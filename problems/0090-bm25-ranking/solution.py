import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    # Preprocessing
    N = len(corpus)  # Total number of documents
    dl = np.array([len(d) for d in corpus])  # Document lengths
    adl = np.mean(dl)  # Average document length

    # Term frequency (TF) calculation
    tf = np.array([[doc.count(q) for q in query] for doc in corpus])

    # Document frequency (DF) calculation
    df = np.array([sum([q in doc for doc in corpus]) for q in query])

    # Inverse Document Frequency (IDF) calculation
    idf = np.log((N + 1) / (df + 1))  # Smoothed IDF

    # BM25 Calculation
    scores = []
    for i, doc in enumerate(corpus):
        score = 0
        for j, q in enumerate(query):
            numerator = tf[i, j] * (k1 + 1)
            denominator = tf[i, j] + k1 * (1 - b + b * dl[i] / adl)
            score += idf[j] * (numerator / denominator)
        scores.append(score)

    return np.round(scores, 3)
