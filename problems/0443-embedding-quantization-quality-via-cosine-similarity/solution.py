import numpy as np

def embedding_quantization_quality(embeddings: np.ndarray, num_bits: int = 8) -> dict:
    """
    Measure the quality of uniform embedding quantization using cosine similarity.
    
    Args:
        embeddings: Input embeddings of shape (n, d)
        num_bits: Number of bits for quantization
    
    Returns:
        Dictionary with mean_cosine_similarity, min_cosine_similarity,
        and max_cosine_similarity (each rounded to 6 decimal places)
    """
    # Your code here
    v_min = embeddings.min(axis=-1, keepdims=True)
    v_max = embeddings.max(axis=-1, keepdims=True)
    L = 2 ** num_bits - 1
    eps = 1e-10
    s = np.maximum((v_max - v_min) / L, eps)
    c = ((embeddings - v_min) / s).round().clip(0, L)
    v_hat = c * s + v_min
    eps = 1e-10
    cos = (embeddings * v_hat).sum(axis=-1) / (eps + np.linalg.norm(embeddings, axis=-1) * np.linalg.norm(v_hat, axis=-1))
    _ = lambda x: round(float(x), 6)
    return {
            "mean_cosine_similarity": _