"""Backbone completo: PatchEmbed + cls token + pos embed + pila de MultiScaleBlock.

Referencia oficial: GraSP/TAPIS/tapis/models/video_model_builder.py::MViT
Verificado contra tapis/config/defaults.py y configs/GraSP/TAPIS/*.yaml.
Usamos una version reducida (menos bloques, dims mas chicas) para poder
iterar rapido en CPU; la mecanica de cada pieza es fiel al original.

CORRECCION (bug de pooling): antes se construian los MultiScaleBlock sin
pasar kernel_q/kernel_kv/stride_kv, asi que tomaban el default (1,1,1) y:
  - pool_q quedaba como una conv de kernel 1 con stride 2 (submuestreo puro,
    no el pooling 3x3x3 de POOL_KVQ_KERNEL), y
  - pool_k / pool_v NUNCA se creaban: K y V no se pooleaban jamas.
Las formas cuadraban igual, por eso ningun test lo detectaba, pero el
modelo no era multiescala. Ver _derive_pooling_schedule mas abajo.
"""
import torch
import torch.nn as nn

from .patch_embed import PatchEmbed
from .block import MultiScaleBlock


def _derive_pooling_schedule(depth, stage_change_layers, kvq_kernel=(3, 3, 3),
                             kv_stride_adaptive=(1, 8, 8), q_stride=(1, 2, 2)):
    """Deriva kernel/stride de pooling de Q y de K/V para cada bloque.

    Replica video_model_builder.py, lineas ~594-635 del repo oficial.

    Q: el kernel (3,3,3) se pone SOLO en las capas que reducen resolucion.
       En las demas, kernel_q queda VACIO -- y eso importa, porque la
       condicion de MultiScaleAttention es
           np.prod(kernel_q) == 1 and np.prod(stride_q) == 1  ->  sin pooling
       y np.prod([]) devuelve 1.0. Si pasaramos (3,3,3) a todas las capas,
       creariamos un pooling con stride 1 en cada bloque, que el original
       no hace.

    K/V: el stride arranca en kv_stride_adaptive y SE VA DIVIDIENDO cada vez
       que una capa poolea Q. Es una variable con estado que persiste entre
       iteraciones, no algo que se calcule por capa:
           [1,8,8] -> [1,4,4] -> [1,2,2] -> [1,1,1]
       La intencion es mantener la resolucion ABSOLUTA de K/V mas o menos
       constante en toda la red: al principio la secuencia es enorme y hay
       que comprimir 8x; al final ya es pequeña y no hace falta comprimir.
       Sin esto, la matriz de atencion de las primeras capas no cabria en
       memoria. Ese es el motivo real de que exista el pooling asimetrico.
       kernel_kv en cambio es (3,3,3) en TODAS las capas, incluidas las
       finales donde el stride ya es [1,1,1]: ahi K/V pasan por una conv
       3x3x3 que suaviza sin reducir.

    Returns:
        kernel_q, stride_q, kernel_kv, stride_kv: listas de longitud `depth`.
        Nota: stride_q vale (1,1,1) -- no () -- en las capas sin pooling,
        porque MultiScaleBlock lo recorre para derivar kernel_skip.
    """
    kernel_q = [() for _ in range(depth)]
    stride_q = [(1, 1, 1) for _ in range(depth)]
    for layer in stage_change_layers:
        kernel_q[layer] = kvq_kernel
        stride_q[layer] = q_stride

    kernel_kv, stride_kv = [], []
    running_kv = list(kv_stride_adaptive)
    for i in range(depth):
        if kernel_q[i]:  # esta capa poolea Q -> K/V necesitan comprimir menos
            running_kv = [max(running_kv[d] // stride_q[i][d], 1) for d in range(3)]
        stride_kv.append(tuple(running_kv))
        kernel_kv.append(kvq_kernel)

    return kernel_q, stride_q, kernel_kv, stride_kv


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
    `SEP_POS_EMBED: True` en los 6 yamls de configs/GraSP/TAPIS/. Hay TRES
    parámetros separados:
        pos_embed_spatial  : (1, H*W, embed_dim)   -- una posición por (h,w)
        pos_embed_temporal : (1, T, embed_dim)     -- una posición por t
        pos_embed_class    : (1, 1, embed_dim)     -- una para el cls token
    y se combinan así (video_model_builder.py, líneas ~858-872):
        pos_embed = pos_embed_spatial.repeat(1, T, 1)
                    + pos_embed_temporal.repeat_interleave(H*W, dim=1)
        pos_embed = cat([pos_embed_class, pos_embed], dim=1)
        x = x + pos_embed
    Funciona porque los tokens estan en orden T-mayor (T mas lento, W mas
    rapido -- el mismo orden de patch_embed.py): `repeat` da
    [hw_0..hw_{HW-1}] repetido T veces, y `repeat_interleave` da
    [t_0]*HW, [t_1]*HW, ... Sumados, el token (t,h,w) recibe
    spatial[h,w] + temporal[t]. Separable usa H*W + T parametros en vez de
    T*H*W y generaliza mejor a otros T.

    LA TRAMPA #3: dim_out y stride_q cambian SOLO en las capas de transición
    de etapa (`stage_change_layers`); en las demás, dim_out == dim y
    stride_q == (1,1,1).

    LA TRAMPA #4 (has_cls_embed=False es un caso real): en
    TAPIS_SHORT/ACTIONS/INSTRUMENTS.yaml, `CLS_EMBED_ON: False` (solo
    LONG/PHASES/STEPS lo dejan en True). Cuando es False no existen
    self.cls_token ni self.pos_embed_class, no se concatena nada, y el
    "resumen global" es x.mean(1) -- replicando lo que hace
    head_helper.py::TransformerBasicHead. Ademas has_cls_embed tiene que
    pasarse a CADA MultiScaleBlock: si no, attention_pool asumiria que el
    token 0 es un cls token y lo excluiria del pooling, corrompiendo la
    rejilla T,H,W.

    Args:
        crop_size: lado espacial H de la entrada.
        crop_size_large: lado W. Si es None, W = H. En el TAPIS real la
            entrada NO es cuadrada en tareas cortas: self.W se calcula con
            TRAIN_CROP_SIZE_LARGE (356 en TAPIS_SHORT.yaml) frente a
            TRAIN_CROP_SIZE (224).
        pool_kvq_kernel: POOL_KVQ_KERNEL del original, (3,3,3).
        pool_kv_stride_adaptive: POOL_KV_STRIDE_ADAPTIVE, (1,8,8) en el real.
            Ojo: con la rejilla de juguete de 16x16 un stride de 8 deja K/V
            en 2x2, que es agresivo. La real arranca de 56x89. Si el modelo
            no aprende nada, prueba (1,4,4) -- pero empieza siendo fiel.
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
        crop_size_large=None,
        has_cls_embed=True,
        pool_kvq_kernel=(3, 3, 3),
        pool_kv_stride_adaptive=(1, 8, 8),
    ):
        super().__init__()
        patch_stride = (2, 4, 4)
        self.patch_embed = PatchEmbed(
            dim_in, embed_dim, kernel=(3, 7, 7), stride=patch_stride, padding=(1, 3, 3)
        )

        if crop_size_large is None:
            crop_size_large = crop_size
        T = num_frames // patch_stride[0]
        H = crop_size // patch_stride[1]
        W = crop_size_large // patch_stride[2]
        self.patch_dims = [T, H, W]

        self.has_cls_embed = has_cls_embed

        if self.has_cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, H * W, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, T, embed_dim))

        nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

        kernel_q, stride_q, kernel_kv, stride_kv = _derive_pooling_schedule(
            depth=depth,
            stage_change_layers=stage_change_layers,
            kvq_kernel=pool_kvq_kernel,
            kv_stride_adaptive=pool_kv_stride_adaptive,
        )

        self.blocks = nn.ModuleList()
        dim_actual = embed_dim
        stage = 0
        for i in range(depth):
            if i in stage_change_layers:
                stage += 1
                dim_out = stage_dims[stage]
            else:
                dim_out = dim_actual

            self.blocks.append(MultiScaleBlock(
                dim=dim_actual,
                dim_out=dim_out,
                num_heads=stage_heads[stage],
                kernel_q=kernel_q[i],
                stride_q=stride_q[i],
                kernel_kv=kernel_kv[i],
                stride_kv=stride_kv[i],
                has_cls_embed=self.has_cls_embed,
            ))
            dim_actual = dim_out

        self.norm = nn.LayerNorm(stage_dims[-1])

    def forward(self, x):
        x, thw = self.patch_embed(x)
        if self.has_cls_embed:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)

        T, H, W = thw
        pos_embed = (self.pos_embed_spatial.repeat(1, T, 1)
                     + self.pos_embed_temporal.repeat_interleave(H * W, dim=1))
        if self.has_cls_embed:
            pos_embed = torch.cat([self.pos_embed_class, pos_embed], dim=1)

        x = x + pos_embed
        for block in self.blocks:
            x, thw = block(x, thw)

        x = self.norm(x)
        if self.has_cls_embed:
            return x[:, 0], x[:, 1:], thw
        else:
            return x.mean(1), x, thw