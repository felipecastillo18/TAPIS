"""Multiscale Pooling Attention (MHPA): el mecanismo central de MViT.

Referencia oficial: GraSP/TAPIS/tapis/models/attention.py::MultiScaleAttention
"""
import torch
import torch.nn as nn


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True):
    """Aplica una capa de pooling 3D (o None) a un tensor de tokens.

    Esta función es pura plomería de reshape: la damos ya resuelta para que
    te concentres en la parte conceptual (MultiScaleAttention de abajo).
    Aun así, LEELA con calma, porque este patrón (separar cls token, volver
    a T,H,W, poolear, aplanar, reponer cls token) se repite en todo MViT.

    Args:
        tensor: (B, num_heads, N, C_head) o (B, N, C).
        pool: nn.Conv3d / nn.AvgPool3d / nn.MaxPool3d, o None.
        thw_shape: [T, H, W] del tensor de entrada (sin contar cls token).
        has_cls_embed: si True, tensor[..., 0, :] es el class token y no se
            poolea.

    Returns:
        tensor poolado, nuevo thw_shape.
    """
    if pool is None:
        return tensor, thw_shape

    tensor_dim = tensor.ndim
    if tensor_dim == 3:
        tensor = tensor.unsqueeze(1)  # (B, 1, N, C) para reusar el mismo código con o sin heads

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    assert L == T * H * W, f"L={L} no coincide con T*H*W={T*H*W}"

    tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    tensor = pool(tensor)
    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)

    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)

    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


class MultiScaleAttention(nn.Module):
    """Self-attention con pooling independiente de Q, K y V.

    Contrato
    --------
    Entrada : x (B, N, dim), thw_shape [T, H, W]  (N incluye cls token si has_cls_embed)
    Salida  : x (B, N_out, dim_out), thw_shape_out

    Idea clave (a diferencia de un ViT normal)
    -------------------------------------------
    En un transformer normal, Q, K y V comparten la misma secuencia de N
    tokens. Aquí, después de proyectar x -> q, k, v, cada uno se poolea por
    SEPARADO con una Conv3d con groups=dim (depthwise) antes de calcular la
    atención:

        q, q_shape = attention_pool(q, self.pool_q, thw_shape)
        k, k_shape = attention_pool(k, self.pool_k, thw_shape)
        v, v_shape = attention_pool(v, self.pool_v, thw_shape)

    Esto es lo que reduce la resolución espacio-temporal a medida que la
    dimensión del canal crece (compute casi constante por etapa). Nota que
    q_shape puede terminar siendo MÁS PEQUEÑO que k_shape/v_shape si
    stride_q > stride_kv: la atención queda "rectangular", no cuadrada,
    y attn.softmax(-1) sigue funcionando igual porque solo opera sobre la
    última dimensión (las keys).

    LA TRAMPA: el número de tokens de salida N_out = prod(q_shape) (+1 si hay
    cls token) casi nunca es igual a N. El caller (MultiScaleBlock) tiene que
    poolear también el residual/shortcut con la MISMA reducción, si no,
    `x_res + attn_out` truena por shapes incompatibles.

    Args:
        dim: canales de entrada.
        dim_out: canales de salida (tras la proyección final `self.proj`).
        num_heads: cabezas de atención (dim_out debe ser divisible).
        kernel_q, stride_q: kernel/stride de la Conv3d depthwise que poolea Q.
        kernel_kv, stride_kv: idem para K y V (K y V comparten stride).
        has_cls_embed: si True, el primer token de x es el class token y se
            excluye del pooling (attention_pool ya lo maneja).
    """

    def __init__(
        self,
        dim,
        dim_out,
        num_heads=8,
        qkv_bias=True,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        has_cls_embed=True,
    ):
        super().__init__()
        raise NotImplementedError(
            "Implementar: self.qkv (Linear dim->3*dim_out), self.proj "
            "(Linear dim_out->dim_out), y self.pool_q/self.pool_k/self.pool_v "
            "(nn.Conv3d depthwise, groups=dim_out//num_heads, o None si "
            "kernel/stride son (1,1,1))."
        )

    def forward(self, x, thw_shape):
        """
        Pasos esperados:
          1. qkv = self.qkv(x) -> reshape a (B, N, 3, num_heads, head_dim) ->
             permute a (3, B, num_heads, N, head_dim) -> separar q, k, v.
          2. Poolear q, k, v con attention_pool(...) usando self.pool_q/k/v.
          3. attn = softmax((q * scale) @ k.transpose(-2, -1), dim=-1)
          4. x = attn @ v
          5. Reordenar de (B, num_heads, N_out, head_dim) a (B, N_out, dim_out)
             y aplicar self.proj.
          6. Devolver x, q_shape (el thw_shape resultante del pooling de Q).
        """
        raise NotImplementedError
