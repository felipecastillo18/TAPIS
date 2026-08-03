"""Cabeza de clasificacion de frame: fases y pasos (tareas largas)."""
import torch.nn as nn


class FrameClassificationHead(nn.Module):
    """Clasifica la ventana temporal completa con la clase del frame central.

    Contrato
    --------
    Entrada : cls_token (B, dim_in)   -- el class token que devuelve mvit.py
              (en el repo oficial esto vive dentro de TransformerBasicHead,
              que recibe la secuencia completa y hace x[:, cls_idx]; aqui
              mvit.py ya separa el cls token, asi que esta cabeza es un
              Linear puro sobre el).
    Salida  : logits (B, num_classes)

    Es literalmente un Linear sobre el class token (mas dropout). En multitarea
    hay una cabeza por tarea y la perdida total es la suma ponderada:

        L = lambda_phases * CE_phases + lambda_steps * CE_steps

    Nota: el paper (Fig. C.17) reporta lambda_phases=1.0, lambda_steps=0.5 como
    el mejor balance, pero TAPIS_LONG.yaml del repo oficial trae [1.0, 1.0].
    Discrepancia sin resolver; a tener en cuenta si intentas reproducir Tabla 9.
    """

    def __init__(self, dim_in=768, num_classes=11, dropout_rate=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes)

    def forward(self, cls_token):
        x = self.dropout(cls_token)
        x = self.projection(x)
        return x
