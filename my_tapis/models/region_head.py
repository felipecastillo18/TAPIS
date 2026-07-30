"""Cabeza de clasificacion de region: instrumentos y acciones (tareas cortas).

Esta es la pieza mas distintiva de TAPIS: en vez de solo pooler el video
globalmente, cruza (cross-attention) los tokens espacio-temporales del video
con los embeddings de cada region/instrumento detectado, para que la
prediccion de "que accion hace este instrumento" tenga contexto de TODO el
clip, no solo del recorte de esa region.

Referencia oficial: GraSP/TAPIS/tapis/models/head_helper.py::TransformerRoIHead
(modo cfg.MODEL.DECODER=True). El original tambien soporta un modo mas
simple (TIME_MLP: mean-pool temporal + concat), que puedes explorar despues
como ejercicio opcional.
"""
import torch
import torch.nn as nn


class RegionClassificationHead(nn.Module):
    """
    Contrato
    --------
    Entradas:
        video_tokens : (B, N, dim_video)      salida de mvit.py (sin cls token)
        region_feats  : (B, R, dim_region)     embeddings de instrumentos
                         (en TAPIS real vienen de Mask2Former; aqui, del
                         dataset sintetico). R = max instrumentos por frame,
                         con padding.
        region_mask   : (B, R) bool            True donde hay un instrumento
                         real (False = padding, se descarta)
    Salida:
        logits : (M, num_classes)   M = numero TOTAL de regiones validas en
                 el batch (region_mask.sum()), aplanado -- no (B, R, C).

    Por que aplanar: cada imagen tiene un numero distinto de instrumentos, y
    solo queremos loss/metricas sobre regiones reales, no sobre el padding.
    El original hace exactamente esto con `x[boxes_mask]` (ver head_helper.py
    linea ~450).

    Mecanismo (modo DECODER):
      1. Proyectar region_feats a la misma dimension que video_tokens
         (dim_region -> dim_video) con una capa lineal.
      2. nn.TransformerDecoderLayer/nn.TransformerDecoder: las regiones
         proyectadas son el "target" (las queries), video_tokens es la
         "memory" (keys/values). Internamente hace primero self-attention
         entre regiones (para que se "vean" entre si) y luego cross-attention
         de cada region hacia los tokens de video.
         Usa tgt_key_padding_mask=~region_mask para que el self-attention
         ignore el padding.
      3. Aplanar con region_mask y proyectar a num_classes.

    LA TRAMPA: nn.TransformerDecoder espera key_padding_mask con True en las
    posiciones a IGNORAR (justo al reves que region_mask, donde True = valido).
    Por eso se le pasa `~region_mask`.
    """

    def __init__(self, dim_video, dim_region, num_classes, num_heads=4,
                 hidden_dim=512, num_layers=2, multi_label=True):
        super().__init__()
        raise NotImplementedError(
            "Implementar: self.region_proj (Linear dim_region->dim_video), "
            "self.decoder (nn.TransformerDecoder con batch_first=True), "
            "self.classifier (Linear dim_video->num_classes). Guardar "
            "multi_label para saber si la loss sera BCE o CrossEntropy."
        )

    def forward(self, video_tokens, region_feats, region_mask):
        raise NotImplementedError
