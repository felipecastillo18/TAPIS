# my_tapis — reimplementación guiada de TAPIS

Objetivo: entender el modelo de `GraSP/TAPIS` escribiendo tú mismo una versión
simplificada, pieza por pieza, comparando contra el código oficial en
`GraSP/TAPIS/tapis/` cuando haga falta.

No reimplementamos toda la infraestructura del repo oficial (SlowFast,
Mask2Former/Detectron2, entrenamiento distribuido). Nos quedamos con la idea
central del paper:

1. Un backbone de video (**MViT**) que produce un *class token* (para tareas
   de frame completo: fase/paso) y una secuencia de tokens espacio-temporales.
2. Un módulo de **regiones** (instrumentos, ya segmentados por otro modelo)
   que aporta un embedding por instancia.
3. Una **cabeza de fusión** que cruza los tokens globales con los embeddings
   de región (atención) para clasificar cada instrumento/acción.
4. Una **cabeza de frame** que clasifica el class token en fase/paso.
5. Una **pérdida multi-tarea** que combina todo.

Cada archivo tiene un docstring con el "Contrato" (forma de entrada/salida) y,
cuando aplica, una nota "LA TRAMPA" señalando el detalle que suele romperse.
Implementa reemplazando el `raise NotImplementedError`. Cuando termines un
archivo, lo revisamos, lo comparamos con el original si aplica, y seguimos
con el siguiente.

## Orden de las etapas

| # | Archivo | Qué enseña | Estado |
|---|---------|------------|--------|
| 1 | `models/patch_embed.py` | Tokenización 3D con Conv3d solapada | ✅ hecho |
| 2 | `models/attention.py` | Multiscale Pooling Attention (MHPA): el corazón de MViT | ✅ hecho |
| 3 | `models/block.py` | Ensamblar attn+MLP con pre-norm y el shortcut "pooled" | ✅ hecho |
| 4 | `models/mvit.py` | Backbone completo: cls token + pos embed + pila de bloques | ✅ hecho |
| 5 | `models/frame_head.py` | Cabeza de fase/paso sobre el class token | ✅ hecho |
| 6 | `models/region_head.py` | La pieza distintiva de TAPIS: fusión región↔global por cross-attention | ✅ hecho |
| 7 | `models/losses.py` | Pérdida multi-tarea ponderada (frame + región) | en curso |
| 8 | `data/toy_dataset.py` | Dataset sintético (sin descargar GraSP) para probar formas | stub |
| 9 | `scripts/train_toy.py` | Loop de entrenamiento end-to-end sobre datos falsos | stub |
| 10 | `scripts/compare_with_official.py` | Verificación final cargando pesos reales de TAPIS | stub |

## Simplificaciones deliberadas (y por qué)

- **No implementamos Mask2Former.** El propio repo oficial entrena TAPIS con
  *features de región precalculadas* (`*_region_features.pth`), no corriendo
  el segmentador en cada forward. Nuestro `toy_dataset.py` genera vectores de
  región falsos del mismo tamaño, así que `region_head.py` es idéntico en
  espíritu al real.
- **No implementamos posiciones relativas (`rel_pos_spatial/temporal`).**
  Son una mejora de MViTv2 sobre MViTv1; el mecanismo de pooling attention
  (la parte conceptualmente nueva) funciona igual sin ellas. Se puede añadir
  después como ejercicio extra.
- **Un solo pathway** (el real soporta SlowFast de 2 pathways vía `dim_in`
  como lista). No lo necesitamos para entender la idea.
- **`region_head.py` implementa solo el modo `DECODER`** (cross-attention),
  que es el mecanismo más representativo del paper. El modo `TIME_MLP`
  (pooling + concat) del original es una alternativa más simple, mencionada
  en el docstring pero no obligatoria.

## Referencia cruzada con el repo oficial

| Nuestro archivo | Original |
|---|---|
| `patch_embed.py` | `tapis/models/stem_helper.py::PatchEmbed` |
| `attention.py` | `tapis/models/attention.py::MultiScaleAttention` |
| `block.py` | `tapis/models/attention.py::MultiScaleBlock` |
| `mvit.py` | `tapis/models/video_model_builder.py::MViT` |
| `frame_head.py` | `tapis/models/head_helper.py::TransformerBasicHead` |
| `region_head.py` | `tapis/models/head_helper.py::TransformerRoIHead` (modo `DECODER`) |
| `losses.py` | `tapis/models/losses.py` |
