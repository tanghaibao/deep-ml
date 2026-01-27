import numpy as np
from typing import List


def hybrid_search_score(
    query_text: str,
    query_embedding: List[float],
    doc_text: str,
    doc_embedding: List[float],
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> float:
    """
    Compute Clawdbot's hybrid search score combining BM25 and vector similarity.

    Args:
        query_text: The search query string
        query_embedding: The query's embedding vector
        doc_text: The document text
        doc_embedding: The document's embedding vector
        bm25_weight: Weight for the BM25 component
        vector_weight: Weight for the vector similarity component

    Returns:
        Combined hybrid score (float)
    """
    bm25 = bm25_score(query_text, doc_text)
    vector_sim = cosine_similarity(query_embedding, doc_embedding)
    hybrid_score = (bm25_weight * bm25) + (vector_weight * vector_sim)
    # print(f"BM25: {bm25}, Vector Similarity: {vector_sim}")
    return hybrid_score


def bm25_score