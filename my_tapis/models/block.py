"""Bloque transformer multiescala: MHPA + MLP con pre-norm."""
import torch.nn as nn

from .attention import MultiScaleAttention, attention_pool


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
      - poolear el shortcut con la MISMA reduccion aplicada a q (usa la
        funcion attention_pool de attention.py, con un nn.MaxPool3d como
        `pool` si stride_q != (1,1,1), o None si no hay cambio de resolucion), y
      - proyectarlo a dim_out con una capa lineal (nn.Linear(dim, dim_out))
        si dim != dim_out.
    Si te lo saltas, el error de formas aparece lejos de aqui y cuesta rastrearlo.

    Sugerencia de __init__:
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads=num_heads,
                                         **attn_kwargs)
        self.norm2 = nn.LayerNorm(dim_out)
        self.mlp = nn.Sequential(nn.Linear(dim_out, int(dim_out*mlp_ratio)), nn.GELU(),
                                  nn.Linear(int(dim_out*mlp_ratio), dim_out))
        self.pool_skip = nn.MaxPool3d(kernel_skip, stride_q, padding_skip) if stride_q != (1,1,1) else None
        self.proj = nn.Linear(dim, dim_out) if dim != dim_out else None
    """

    def __init__(self, dim, dim_out, num_heads, mlp_ratio=4.0,
                 stride_q=(1, 1, 1), **attn_kwargs):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, thw):
        """
        Pasos esperados:
          1. x_norm = self.norm1(x)
          2. x_attn, thw_out = self.attn(x_norm, thw)
          3. x_res, _ = attention_pool(x, self.pool_skip, thw, has_cls_embed=True)
             (x_res es x SIN normalizar, pooleado a la nueva resolucion)
          4. if self.proj is not None: x_res = self.proj(x_res)
          5. x = x_res + x_attn   (sin drop_path por simplicidad)
          6. x = x + self.mlp(self.norm2(x))
          7. return x, thw_out
        """
        raise NotImplementedError
