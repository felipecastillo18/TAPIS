"""Bloque transformer multiescala: MHPA + MLP con pre-norm."""
import torch.nn as nn


class MultiScaleBlock(nn.Module):
    """Bloque de MViT.

    Contrato
    --------
    Entrada : x (B, N, dim), thw [T,H,W]
    Salida  : x (B, N_out, dim_out), thw_out

    Estructura (pre-norm, como ViT):
        x = shortcut + drop_path(attn(norm1(x)))
        x = x        + drop_path(mlp(norm2(x)))

    LA TRAMPA: cuando el bloque cambia de resolucion (stride_q != 1) o de
    dimension (dim != dim_out), el atajo residual NO puede ser x directo.
    Hay que:
      - poolear el shortcut con la misma reduccion aplicada a q, y
      - proyectarlo a dim_out con una capa lineal.
    Si te lo saltas, el error de formas aparece lejos de aqui y cuesta rastrearlo.
    """

    def __init__(self, dim, dim_out, num_heads, mlp_ratio=4.0,
                 drop_path=0.0, **attn_kwargs):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, thw):
        raise NotImplementedError
