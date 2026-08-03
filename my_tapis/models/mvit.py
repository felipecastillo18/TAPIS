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

    LA TRAMPA #4 (has_cls_embed=False es un caso real, no un adorno): en
    TAPIS_SHORT/ACTIONS/INSTRUMENTS.yaml, `CLS_EMBED_ON: False` (solo
    LONG/PHASES/STEPS lo dejan en True). Verificado en video_model_builder.py
    líneas 553-557 (init) y 850-856 (forward): cuando es False, **no existe**
    `self.cls_token` ni `self.pos_embed_class` -- no se crean, no se
    concatena nada, la secuencia se queda siendo solo los patches. Y en
    head_helper.py::TransformerBasicHead.forward (ya visto en frame_head.py),
    cuando `cls_embed=False` la cabeza de frame usa `x.mean(1)` (promedio de
    TODOS los tokens) en vez de leer un token dedicado.
    Para mantener el contrato de esta clase estable (siempre devolver un
    `cls_token` de forma `(B, dim_final)`, sin importar el flag), cuando
    `has_cls_embed=False` debes:
      - NO crear self.cls_token ni self.pos_embed_class.
      - NO concatenar nada extra a la secuencia (x se queda con solo patches).
      - Al final, devolver `x.mean(1)` como "cls_token" (replicando lo que
        hace TransformerBasicHead), y `x` completo (sin recortar el token 0,
        porque no hay ningún token 0 especial) como `tokens`.
    Ademas, `has_cls_embed` tiene que pasarse a CADA `MultiScaleBlock` (que a
    su vez lo pasa a `MultiScaleAttention` y a `attention_pool`): si no,
    `attention_pool` asumiria por default que el token 0 es un cls token y
    lo excluiria del pooling espacial, corrompiendo la rejilla T,H,W cuando
    en realidad ese token 0 es un patch normal.
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
        self.patch_embed = PatchEmbed(dim_in, embed_dim, kernel = (3, 7, 7), stride=(2, 4, 4),
                                    padding=(1, 3, 3))
        T = num_frames // 2
        H = crop_size // 4
        W = H
        patch_dims = [T, H, W]

        self.has_cls_embed = has_cls_embed

        if self.has_cls_embed:

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, H*W, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, T, embed_dim))

        nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

        self.blocks = nn.ModuleList()
        dim_actual = embed_dim
        stage = 0
        for i in range(depth):
            if i in stage_change_layers:
                stage += 1
                dim_out = stage_dims[stage]
                stride_q = (1, 2, 2)
            else:
                dim_out = dim_actual
                stride_q = (1, 1, 1)

            self.blocks.append(MultiScaleBlock(
                dim=dim_actual,
                dim_out=dim_out,
                num_heads=stage_heads[stage],
                stride_q=stride_q,
                has_cls_embed=self.has_cls_embed,
            ))
            dim_actual = dim_out

        self.norm = nn.LayerNorm(stage_dims[-1])


    def forward(self, x):
        """
        Pasos esperados:
          1. x, thw = self.patch_embed(x)
          2. SOLO SI self.has_cls_embed:
                cls = self.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls, x], dim=1)
          3. T, H, W = thw
             pos_embed = self.pos_embed_spatial.repeat(1, T, 1) \\
                 + self.pos_embed_temporal.repeat_interleave(H * W, dim=1)
             SOLO SI self.has_cls_embed:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], dim=1)
             x = x + pos_embed
          4. for block in self.blocks: x, thw = block(x, thw)
          5. x = self.norm(x)
          6. SI self.has_cls_embed: return x[:, 0], x[:, 1:], thw
             SI NO:                 return x.mean(1), x, thw
             (ver LA TRAMPA #4: sin cls token, "el resumen global" es el
             promedio de todos los tokens, y no hay ningun token que recortar
             de `tokens`)
        """
        x, thw = self.patch_embed(x)
        if self.has_cls_embed:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim = 1)

        T, H, W = thw
        pos_embed = self.pos_embed_spatial.repeat(1, T, 1) + self.pos_embed_temporal.repeat_interleave(H * W, dim = 1)
        for block in self.blocks:
            x, thw = block(x, thw)

        x = self.norm(x)
        if self.has_cls_embed:
            return x[:, 0], x[:, 1:], thw
        else:
            return x.mean(1), x, thw
