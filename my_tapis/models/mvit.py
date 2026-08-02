"""Backbone completo: PatchEmbed + cls token + pos embed + pila de MultiScaleBlock.

Referencia oficial: GraSP/TAPIS/tapis/models/video_model_builder.py::MViT
Verificado contra tapis/config/defaults.py y configs/GraSP/TAPIS/*.yaml.
Usamos una version reducida (menos bloques, dims mas chicas) para poder
iterar rapido en CPU; la mecanica de cada pieza es fiel al original.
"""
import torch
import torch.nn as nn

from .patch_embed import PatchEmbed
from .block import MultiScaleBlock


class MViT(nn.Module):
    """
    Contrato
    --------
    Entrada : clip (B, 3, T, H, W)   default (B, 3, 8, 64, 64) -- mismo
              tamaño que genera data/toy_dataset.py.
    Salida  : cls_token (B, dim_final), tokens (B, N, dim_final), thw_shape

    cls_token es el resumen global del clip (lo usa frame_head.py). tokens es
    la secuencia espacio-temporal completa (SIN el cls token), memoria para
    el cross-attention de region_head.py.

    Arquitectura de juguete (inspirada en la real: embed_dim=96, DEPTH=16,
    DIM_MUL en capas [1,3,14] -> 96,192,384,768; aqui usamos solo 2
    transiciones para que corra rapido en CPU):
        depth=6, stage_dims=(96,192,384), stage_heads=(1,2,4),
        stage_change_layers=(1,3)  # stride_q=(1,2,2) en esas 2 capas

    LA TRAMPA #1 (cls token): NO pasa por PatchEmbed. Es un
    nn.Parameter(torch.zeros(1, 1, embed_dim)) que se expande al batch y se
    concatena AL INICIO de la secuencia de patches, DESPUÉS de correr
    PatchEmbed pero ANTES de sumar el positional embedding.

    LA TRAMPA #2 (positional embedding SEPARABLE, no uno solo): en TAPIS,
    `SEP_POS_EMBED: True` en los 6 yamls de configs/GraSP/TAPIS/ (el default
    de defaults.py es False, pero ningún config real lo usa). Esto significa
    que NO hay un solo `nn.Parameter(1, 1+T*H*W, embed_dim)`. Hay TRES
    parámetros separados:
        pos_embed_spatial  : (1, H*W, embed_dim)   -- una posición por (h,w)
        pos_embed_temporal : (1, T, embed_dim)     -- una posición por t
        pos_embed_class    : (1, 1, embed_dim)     -- una para el cls token
    y se combinan así (video_model_builder.py, líneas ~858-872):
        pos_embed = pos_embed_spatial.repeat(1, T, 1)
                    + pos_embed_temporal.repeat_interleave(H*W, dim=1)
        pos_embed = cat([pos_embed_class, pos_embed], dim=1)
        x = x + pos_embed
    Por qué funciona el `repeat`/`repeat_interleave`: tus tokens están en
    orden T-mayor (T más lento, W más rápido -- el mismo orden de
    patch_embed.py). `pos_embed_spatial.repeat(1, T, 1)` da
    `[hw_0..hw_{HW-1}, hw_0..hw_{HW-1}, ...]` (T veces), y
    `pos_embed_temporal.repeat_interleave(H*W, dim=1)` da
    `[t_0]*HW, [t_1]*HW, ...` -- exactamente ese mismo orden. Sumados,
    token (t,h,w) recibe `spatial[h,w] + temporal[t]`.
    Por qué separable y no una tabla completa T*H*W: son muchos menos
    parámetros (`H*W + T` en vez de `T*H*W`) y generaliza mejor a otros T.
    (El original también soporta interpolar el pos_embed si el tamaño de
    entrada cambia entre train/test -- lo omitimos: nuestro MViT de juguete
    siempre recibe el mismo T,H,W con el que fue construido.)

    LA TRAMPA #3: en cada MultiScaleBlock nuevo, dim_out cambia SOLO en las
    capas de transición de etapa (`stage_change_layers`); en las demás,
    dim_out == dim. Lo mismo aplica a stride_q: es (1,2,2) solo en esas
    capas, (1,1,1) en el resto.
    """

    def __init__(
        self,
        dim_in=3,
        embed_dim=96,
        depth=6,
        stage_dims=(96, 192, 384),
        stage_heads=(1, 2, 4),
        stage_change_layers=(1, 3),
        num_frames=8,
        crop_size=64,
        has_cls_embed=True,
    ):
        super().__init__()
        raise NotImplementedError(
            "Implementar:\n"
            "1. self.patch_embed = PatchEmbed(dim_in, embed_dim, "
            "kernel=(3,7,7), stride=(2,4,4), padding=(1,3,3))\n"
            "2. Calcular patch_dims = [T,H,W] resultantes SIN correr nada: "
            "para este kernel/stride/padding, T=num_frames//2, "
            "H=W=crop_size//4 (division entera -- coincide exacto con la "
            "formula de conv de patch_embed.py, verificalo si quieres).\n"
            "3. self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))\n"
            "4. self.pos_embed_spatial = nn.Parameter(torch.zeros(1, H*W, embed_dim))\n"
            "   self.pos_embed_temporal = nn.Parameter(torch.zeros(1, T, embed_dim))\n"
            "   self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))\n"
            "   (nn.init.trunc_normal_(p, std=0.02) sobre los tres es buena "
            "practica, no imprescindible para que el forward corra)\n"
            "5. self.blocks = nn.ModuleList(...): recorrer range(depth), "
            "llevando dim_actual (arranca en embed_dim) y un indice de etapa "
            "(arranca en 0). En cada capa i: dim_out = stage_dims[etapa+1] "
            "si i esta en stage_change_layers, si no dim_out = dim_actual. "
            "stride_q = (1,2,2) si i esta en stage_change_layers, si no "
            "(1,1,1). num_heads = stage_heads[etapa] (o stage_heads[etapa+1] "
            "tras la transicion -- se consistente). Actualizar dim_actual = "
            "dim_out despues de cada bloque, y avanzar la etapa cuando "
            "corresponda.\n"
            "6. self.norm = nn.LayerNorm(stage_dims[-1])"
        )

    def forward(self, x):
        """
        Pasos esperados:
          1. x, thw = self.patch_embed(x)
          2. cls = self.cls_token.expand(x.shape[0], -1, -1)
             x = torch.cat([cls, x], dim=1)
          3. T, H, W = thw
             pos_embed = self.pos_embed_spatial.repeat(1, T, 1) \\
                 + self.pos_embed_temporal.repeat_interleave(H * W, dim=1)
             pos_embed = torch.cat([self.pos_embed_class, pos_embed], dim=1)
             x = x + pos_embed
          4. for block in self.blocks: x, thw = block(x, thw)
          5. x = self.norm(x)
          6. return x[:, 0], x[:, 1:], thw   # cls_token, tokens, thw_shape
        """
        raise NotImplementedError
