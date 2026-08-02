"""Bloque transformer multiescala: MHPA + MLP con pre-norm.

Referencia oficial: GraSP/TAPIS/tapis/models/attention.py::MultiScaleBlock
Verificado contra tapis/config/defaults.py y configs/GraSP/TAPIS/*.yaml.
"""
import torch.nn as nn

from .attention import MultiScaleAttention, attention_pool


class MultiScaleBlock(nn.Module):
    """Bloque de MViT.

    Contrato
    --------
    Entrada : x (B, N, dim), thw [T,H,W]
    Salida  : x (B, N_out, dim_out), thw_out

    Estructura (pre-norm, como ViT):
        x = shortcut(x)          + drop_path(attn(norm1(x)))
        x = shortcut(norm2(...)) + drop_path(mlp(norm2(x)))

    LA TRAMPA #1 (quién expande los canales): en TAPIS, `MultiScaleAttention`
    NUNCA cambia el ancho del canal. Se ve tentador construirla como
    `MultiScaleAttention(dim, dim_out, ...)`, pero eso es incorrecto: hay que
    construirla como `MultiScaleAttention(dim, dim, ...)`. El cambio de
    `dim -> dim_out` lo hace EXCLUSIVAMENTE el MLP (`nn.Linear(dim, hidden)`
    seguido de `nn.Linear(hidden, dim_out)`) y una proyección lineal aparte
    para el residual (`self.proj`). Esto es así porque `cfg.MVIT.DIM_MUL_IN_ATT`
    vale `False` en TAPIS (default en defaults.py, sin override en ningún yaml
    de configs/GraSP/) — el caso `True` existe en el código general de MViT
    pero no se usa aquí, así que no lo implementamos.
    Consecuencia: `self.norm2` normaliza sobre `dim` (no `dim_out`), porque
    normaliza la salida de la atención, que todavía no cambió de ancho.

    LA TRAMPA #2 (el shortcut real, no el intuitivo): cuando el bloque cambia
    de resolucion (stride_q reduce H,W) o de dimension (dim != dim_out), hay
    DOS residuales distintos, en momentos distintos:
      - El residual de la atención (`x_res`) es la entrada ORIGINAL `x`
        (no normalizada, no proyectada) pooleada con la MISMA reduccion que
        se le aplico a Q. Nunca se proyecta aqui, porque dim_mul_in_att=False
        implica que la atencion no cambio el ancho, asi que no hay nada que
        proyectar todavia.
      - El residual del MLP, en cambio, SI se proyecta -- pero ojo: se
        proyecta `norm2(x)` (la version YA NORMALIZADA), no `x` cruda. Es
        decir, en los 3 bloques de TAPIS donde `dim != dim_out`, el camino
        residual pasa por un LayerNorm antes de la proyeccion lineal. Esto
        parece un bug (normalizar el residual no es lo tipico) pero es
        exactamente lo que hace el MViT oficial -- no lo "arregles".

    LA TRAMPA #3 (con qué thw se poolea el shortcut): `attention_pool` para
    el shortcut de la atencion recibe el thw de ENTRADA (`thw`, el mismo que
    recibio el bloque), NO el thw ya reducido que devuelve `self.attn`. Tiene
    sentido: el tensor que estas pooleando (`x`, la entrada original) todavia
    tiene la resolucion de entrada, asi que `attention_pool` necesita saber
    esa resolucion para desplegarlo correctamente, sin importar a qué
    resolucion lo vas a reducir.

    LA TRAMPA #4 (cuándo existe el pool_skip): la condicion NO es
    `stride_q != (1,1,1)` (eso compara una lista contra una tupla y nunca da
    True). Es `len(stride_q) > 0 and prod(stride_q) > 1`. `kernel_skip` y
    `padding_skip` se derivan de `stride_q`, no son un parametro aparte:
        kernel_skip  = [s + 1 if s > 1 else s for s in stride_q]
        padding_skip = [k // 2 for k in kernel_skip]
    (con stride_q=(1,2,2) esto da kernel=[1,3,3], padding=[0,1,1] -- el mismo
    tamaño de salida que produce la Conv3d que poolea Q dentro de la atención).

    Nota sobre `has_cls_embed`: NO se puede hardcodear en `True`. Varía por
    tarea: `TAPIS_LONG/PHASES/STEPS.yaml` traen `CLS_EMBED_ON: True`, pero
    `TAPIS_SHORT/ACTIONS/INSTRUMENTS.yaml` traen `CLS_EMBED_ON: False`. Debe
    ser un argumento del constructor.

    Deuda técnica explícita (no implementada aquí): el `drop_path`
    (stochastic depth) se omite -- el forward no lo aplica, aunque el
    parametro se reciba. Esto NO es una simplificación inocua: en TAPIS,
    `DROPPATH_RATE` vale 0.4 en las tareas cortas (SHORT/ACTIONS/INSTRUMENTS)
    y 0.2 en las largas (LONG/PHASES/STEPS) -- es el regularizador principal
    del modelo. Si en algún momento este bloque no generaliza bien
    entrenando con datos reales, este es el primer sospechoso a revisar.
    (gamma_1/gamma_2 de layer-scale sí se omiten sin pérdida real: no existe
    ningún campo de config para layer_scale_init_value, y el call site en
    video_model_builder.py nunca lo pasa, así que en este repo siempre vale
    su default 0.0 -- omitirlos es fiel al comportamiento real, no una
    simplificación.)

    Args:
        dim: canales de entrada.
        dim_out: canales de salida (puede ser igual a dim, o mayor en los
            3 bloques de transición de etapa).
        num_heads: cabezas de atención, pasado a MultiScaleAttention.
        mlp_ratio: factor de expansión del MLP (hidden = dim * mlp_ratio).
        kernel_q, kernel_kv, stride_q, stride_kv: pooling de Q/K/V, pasado
            tal cual a MultiScaleAttention. stride_q también determina
            kernel_skip/pool_skip aquí.
        has_cls_embed: si el primer token de x es un class token (varía
            por tarea, ver nota arriba). Se pasa también a MultiScaleAttention.
        drop_path: guardado pero no aplicado (ver deuda técnica arriba).
    """

    def __init__(self, dim, dim_out, num_heads, mlp_ratio=4.0,
                 kernel_q=(1, 1, 1), kernel_kv=(1, 1, 1),
                 stride_q=(1, 1, 1), stride_kv=(1, 1, 1),
                 has_cls_embed=True, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.has_cls_embed = has_cls_embed

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiScaleAttention(dim, dim, num_heads=num_heads, 
                                        kernel_q=kernel_q, kernel_kv=kernel_kv, stride_q=stride_q, stride_kv=stride_kv,
                                        has_cls_embed=has_cls_embed)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim_out)
        )

        self.proj = nn.Linear(dim, dim_out) if dim != dim_out else None

        kernel_skip = [s+1 if s>1 else s for s in stride_q]
        padding_skip = [k//2 for k in kernel_skip]

        self.pool_skip = nn.MaxPool3d(kernel_skip, stride_q, padding_skip) if len(stride_q) > 0 and __import__('numpy').prod(stride_q) > 1 else None

    def forward(self, x, thw):
        """
        Pasos esperados (fieles al MViT oficial, ver trampas #2 y #3 arriba):
          1. x_norm = self.norm1(x)
          2. x_block, thw_new = self.attn(x_norm, thw)
          3. x_res, _ = attention_pool(x, self.pool_skip, thw,
                                        has_cls_embed=self.has_cls_embed)
             (nota: pooleamos `x`, NO `x_norm`; y con `thw`, NO `thw_new`)
          4. x = x_res + x_block   (sin drop_path, ver deuda técnica)
          5. x_norm = self.norm2(x)
          6. x_mlp = self.mlp(x_norm)
          7. if self.proj is not None: x = self.proj(x_norm)
             (nota: se proyecta x_norm, NO el x del paso 4 -- ver trampa #2)
          8. x = x + x_mlp
          9. return x, thw_new
        """
        x_norm = self.norm1(x)
        x_block, thw_new = self.attn(x_norm, thw)
        x_res, _ = attention_pool(x, self.pool_skip, thw, has_cls_embed=self.has_cls_embed)
        x = x_res + x_block
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.proj is not None:
            x = self.proj(x_norm)
        x += x_mlp
        return x, thw_new
