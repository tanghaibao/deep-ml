import numpy as np
from collections import Counter

def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """
    Calculate BLEU score for a candidate sentence against reference sentences.
    
    Args:
        candidate: List of tokens in the candidate sentence
        references: List of reference sentences, each as a list of tokens
        max_n: Maximum n-gram order (default: 4)
    
    Returns:
        BLEU score between 0 and 1
    """
    # Your code here
    p = np.zeros(max_n)
    c = len(candidate)
    _, r = min((abs(len(ref) - c), len(ref)) for ref in references)
    bp = 1 if c >= r else np.exp(1 - r / c)
    def ngram(L, k):
        res = []
        for i in range(len(L) - k + 1):
            res.append(tuple(L[i: i + k]))
        return Counter(res)
    for k in range(1, max_n + 1):
        ng_cand = ngram(candidate, k)
        ng_refs = [ngram(ref, k) for ref in references]
        a = 0
        for ng, v in ng_cand.items():
 