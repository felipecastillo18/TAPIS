"""Loop de entrenamiento end-to-end sobre datos sinteticos.

Este es el "examen final" de las piezas del backbone: si esto corre y la
loss baja de forma consistente en unos pocos cientos de pasos sobre datos
aleatorios, tu MViT + frame_head + region_head + losses estan bien
conectados. No demuestra que el modelo "aprenda algo real" (los datos son
ruido), solo que las formas y gradientes fluyen correctamente.

Referencia oficial: no hay un analogo directo simple; el loop real esta
repartido entre tools/train_net.py y tapis/models/video_model_builder.py
del repo GraSP/TAPIS.
"""
import torch
from torch.utils.data import DataLoader

from my_tapis.data.toy_dataset import ToyGraspDataset
from my_tapis.models.mvit import MViT
from my_tapis.models.frame_head import FrameClassificationHead
from my_tapis.models.region_head import RegionClassificationHead
from my_tapis.models.losses import TapisMultiTaskLoss


def main():
    """
    Pasos esperados:
      1. dataset = ToyGraspDataset(...); loader = DataLoader(dataset, batch_size=4)
      2. backbone = MViT(...)
      3. frame_head = FrameClassificationHead(dim_in=<dim_final_de_mvit>, num_classes=num_phases)
      4. region_head = RegionClassificationHead(dim_video=<dim_final_de_mvit>,
             dim_region=dim_region, num_classes=num_instruments)
      5. loss_fn = TapisMultiTaskLoss(
             task_losses={"phases": "cross_entropy", "instruments": "bce_logit"},
             task_weights={"phases": 1.0, "instruments": 1.0})
      6. optimizer = torch.optim.Adam(list(backbone.parameters()) +
             list(frame_head.parameters()) + list(region_head.parameters()), lr=1e-4)
      7. Loop de N pasos:
           - cls_token, tokens, thw = backbone(batch["clip"])
           - phase_logits = frame_head(cls_token)
           - instr_logits = region_head(tokens, batch["region_feats"], batch["region_mask"])
           - instr_targets = batch["instruments"][batch["region_mask"]]  # aplanar igual que instr_logits
           - loss, parts = loss_fn(
                 {"phases": phase_logits, "instruments": instr_logits},
                 {"phases": batch["phase"], "instruments": instr_targets})
           - loss.backward(); optimizer.step(); optimizer.zero_grad()
           - imprimir loss cada N pasos y verificar que baja
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
