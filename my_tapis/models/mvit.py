"""Backbone completo: PatchEmbed + cls token + pos embed + pila de MultiScaleBlock.

Referencia oficial: GraSP/TAPIS/tapis/models/video_model_builder.py::MViT
(usamos una version reducida; el original tiene DEPTH=16 para GraSP, aqui
usamos algo mas chico para poder iterar rapido en CPU).
"""
import torch
import torch.nn as nn

from .patch_embed import PatchEmbed
from .block import MultiScaleBlock


class MViT(nn.Module):
    """
    Contrato
    --------
    Entrada : clip (B, 3, T, H, W)   p.ej. (B, 3, 16, 224, 224)
    Salida  : cls_token (B, dim_final), tokens (B, N, dim_final), thw_shape

    cls_token es el resumen global del clip (lo usa frame_head.py para
    clasificar fase/paso). tokens es la secuencia espacio-temporal completa
    (SIN el cls token) que usa region_head.py como memoria para cross-attention.

    Arquitectura (config de juguete, inspirada en la real de GraSP):
        embed_dim inicial = 96
        3 etapas: dim 96 -> 192 -> 384, heads 1 -> 2 -> 4
        el cambio de etapa ocurre en las capas donde stride_q reduce H,W a la mitad
        ejemplo con depth=6: stride_q=(1,2,2) en las capas 1 y 3 (0-indexed)

    LA TRAMPA #1: el cls_token NO pasa por PatchEmbed. Se crea como
    nn.Parameter(torch.zeros(1, 1, embed_dim)) y se concatena (repetido por
    batch) al inicio de la secuencia de patches ANTES del primer bloque.

    LA TRAMPA #2: el positional embedding tiene que cubrir cls_token + todos
    los patches: nn.Parameter(torch.zeros(1, 1 + T*H*W, embed_dim)), y se
    suma (no concatena) justo despues de armar la secuencia con el cls token.
    Su tamaño esta fijo al T,H,W de PatchEmbed (no al de las etapas
    posteriores, que van encogiendo).

    LA TRAMPA #3: en cada MultiScaleBlock nuevo, dim_out cambia solo en las
    capas de transicion de etapa; en las demas capas dim_out == dim.
    """

    def __init__(
        self,
        dim_in=3,
        embed_dim=96,
        depth=6,
        stage_dims=(96, 192, 384),
        stage_heads=(1, 2, 4),
        stage_change_layers=(1, 3),  # indices de capa donde sube la etapa
        num_frames=16,
        crop_size=224,
    ):
        super().__init__()
        raise NotImplementedError(
            "Implementar: self.patch_embed, self.cls_token, self.pos_embed, "
            "self.blocks (nn.ModuleList de MultiScaleBlock, con dim/dim_out/"
            "num_heads/stride_q segun la etapa de cada capa), self.norm final."
        )

    def forward(self, x):
        """
        Pasos esperados:
          1. x, thw = self.patch_embed(x)                     # (B, N, embed_dim)
          2. cls = self.cls_token.expand(x.shape[0], -1, -1)
             x = torch.cat([cls, x], dim=1)
          3. x = x + self.pos_embed
          4. for block in self.blocks: x, thw = block(x, thw)
          5. x = self.norm(x)
          6. return x[:, 0], x[:, 1:], thw   # cls_token, tokens, thw_shape
        """
        raise NotImplementedError
