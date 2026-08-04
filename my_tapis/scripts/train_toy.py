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
    dataset = ToyGraspDataset()
    loader = DataLoader(dataset=dataset, batch_size=4)

    backbone = MViT()
    frame_head = FrameClassificationHead(dim_in=384, num_classes=dataset.num_phases)
    region_head = RegionClassificationHead(
        dim_video=384,
        dim_region=dataset.dim_region,
        num_classes=dataset.num_instruments,
    )
    loss_fn = TapisMultiTaskLoss(
        task_losses={"phases": "cross_entropy", "instruments": "bce_logit"},
        task_weights={"phases": 1.0, "instruments": 1.0},
    )
    optimizer = torch.optim.Adam(
        list(backbone.parameters())
        + list(frame_head.parameters())
        + list(region_head.parameters()),
        lr=1e-4,
    )

    step = 0
    for epoch in range(5):
        for batch in loader:
            cls_token, tokens, thw = backbone(batch["clip"])
            phase_logits = frame_head(cls_token)
            instr_logits = region_head(tokens, batch["region_feats"], batch["region_mask"])
            instr_targets = batch["instruments"][batch["region_mask"]]

            loss, parts = loss_fn(
                {"phases": phase_logits, "instruments": instr_logits},
                {"phases": batch["phase"], "instruments": instr_targets},
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            print(f"step {step:3d}  loss={loss.item():.4f}  "
                  f"phases={parts['phases'].item():.4f}  "
                  f"instruments={parts['instruments'].item():.4f}")


if __name__ == "__main__":
    main()
