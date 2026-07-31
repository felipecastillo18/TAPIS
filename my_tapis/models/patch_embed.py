"""Patchificacion 3D de MViT.

A diferencia de ViT, los patches se SOLAPAN: kernel (3,7,7) con stride (2,4,4).
Eso es una Conv3d, no un reshape.
"""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Convierte un clip en una secuencia de tokens espacio-temporales.

    Contrato
    --------
    Entrada : x    (B, C_in, T, H, W)
    Salida  : x    (B, N, C_out)   con N = T_out * H_out * W_out
              thw  [T_out, H_out, W_out]

    Con la config de TAPIS (kernel=(3,7,7), stride=(2,4,4), padding=(1,3,3))
    y entrada (B, 3, 16, 224, 224) -> N = 8 * 56 * 56 = 25088, C_out = 96.

    Nota: el orden del flatten importa. Debe ser (T, H, W) con W variando mas
    rapido, porque todo el resto del modelo (pos embeds, pooling, reshapes en
    MHPA) asume ese layout. Si te equivocas aqui nada falla ruidosamente:
    simplemente el modelo no aprende bien.
    """

    def __init__(self, dim_in=3, dim_out=96, kernel=(3, 7, 7),
                 stride=(2, 4, 4), padding=(1, 3, 3)):
        super().__init__()

        self.conv = nn.Conv3d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor):

        res_conv = self.conv(x)
        thw = [res_conv.shape[2], res_conv.shape[3], res_conv.shape[4]]
        x_final = torch.Tensor.flatten(res_conv, 2 , -1).transpose(1, 2) # Solo quiero aplanar las últimas 3 coordenadas de mi tensor. (temporal, w, h)
        return x_final, thw