import numpy as np

def softmax(x):
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def rope(X, d, base, positions):
    theta = 1 / base ** (2 * np.arange(d // 2) / d)
    mt = (positions[:, None] * theta[None, :]).reshape(-1)
    a = np.cos(mt)  # (positions, d / 2, 1)
    b = np.sin(mt)
    p, = positions.shape
    r = np.concatenate([a[:, None], -b[:, None], b[:, None], a[:, None]], axis=-1) 
    r = r.reshape((p, d // 2, 2, 2))
    x = X.reshape((p, -1, 2, 1))
    x = r @ x
    x = x.reshape((p, d))
    return x


def irope_attention(Q: list, K: list, V: list, positions: list, layer_index: int, rope_layers: list, base: float = 10000.0) -> dict:
    """
    Compute attention for a single layer in an iRoPE transformer.
    
    Args:
        Q: Query matrix, shape (seq_len, d_head)
        K: Key matrix, shape (seq_len, d_head)
        V: Value matrix, shape (seq_len, d_head)
        positions: Position indices, shape (seq_len,)
  