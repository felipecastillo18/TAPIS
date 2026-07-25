"""Compara tu implementacion contra la oficial cargando los mismos pesos.

Este es el test definitivo: si las salidas coinciden numericamente, tu
reimplementacion es fiel. Si no, la diferencia te dice donde buscar.

Uso:
    python scripts/compare_with_official.py --ckpt /ruta/al/checkpoint.pyth

Estrategia:
  1. Cargar el state_dict oficial.
  2. Instanciar tu modelo y llamar load_state_dict(..., strict=True).
     Si falla, el mensaje te lista exactamente que parametros no cuadran:
     esa lista ES tu lista de tareas pendientes.
  3. Pasar el mismo tensor de entrada por ambos y comparar con torch.allclose.
  4. Si no coinciden, comparar capa por capa con hooks forward para localizar
     el primer punto de divergencia.
"""
import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state", ckpt)

    print(f"{len(state)} tensores en el checkpoint. Primeras claves:")
    for k in list(state)[:20]:
        print(f"  {k:60s} {tuple(state[k].shape)}")

    # TODO: instanciar tu MViT y hacer load_state_dict(state, strict=True)
    raise NotImplementedError("Implementar cuando MViT este listo")


if __name__ == "__main__":
    main()
