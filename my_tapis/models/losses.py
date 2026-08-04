"""Perdida multi-tarea ponderada: combina tareas de frame y de region.

Referencia oficial: GraSP/TAPIS/tapis/models/losses.py
(el original es basicamente esto: un registro de nn.CrossEntropyLoss /
nn.BCEWithLogitsLoss por nombre, mas una suma ponderada por lambdas de config).
"""
import torch.nn as nn

_LOSS_CLASSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce_logit": nn.BCEWithLogitsLoss,
}

class TapisMultiTaskLoss(nn.Module):
    """
    Contrato
    --------
    Entrada : predictions: dict[str, Tensor], targets: dict[str, Tensor]
              con las mismas claves. Por ejemplo:
                predictions = {"phases": (B, 11), "instruments": (M, 7)}
                targets     = {"phases": (B,) long, "instruments": (M, 7) float}
              (M = numero de regiones validas en el batch, ver region_head.py)
    Salida  : loss_total (escalar), dict con cada loss individual (para logging)

    Cada tarea tiene:
      - una funcion de perdida: CrossEntropyLoss (multi-clase, un instrumento
        activo a la vez, ej. fase/paso) o BCEWithLogitsLoss (multi-label,
        varias acciones pueden estar activas a la vez para un instrumento)
      - un peso lambda (cfg.TASKS.LOSS_WEIGHTS en el original)

        L_total = sum_t  lambda_t * L_t(predictions[t], targets[t])

    LA TRAMPA: BCEWithLogitsLoss espera logits crudos (sin sigmoid aplicado),
    mientras que CrossEntropyLoss espera logits crudos tambien pero targets
    como indices de clase (long), no one-hot. Si tus heads ya aplican softmax/
    sigmoid en forward (como hace el original en eval), tienes que
    desactivarlo durante training o usar la version "logit" de la perdida.
    """

    def __init__(self, task_losses, task_weights):
        """
        Args:
            task_losses: dict[str, str], p.ej. {"phases": "cross_entropy",
                "instruments": "bce_logit"}.
            task_weights: dict[str, float], p.ej. {"phases": 1.0,
                "instruments": 1.0}.
        """
        super().__init__()

        self.losses = nn.ModuleDict({
            task: _LOSS_CLASSES[loss_name]()
            for task, loss_name in task_losses.items()
        })
        self.task_weights = task_weights


    def forward(self, predictions, targets):
        parts = {}
        for task, loss_fn in self.losses.items():
            parts[task] = loss_fn(predictions[task], targets[task])

        total = sum(self.task_weights[task] * parts[task] for task in parts)
        return total, parts