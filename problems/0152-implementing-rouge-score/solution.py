# Implement your function below.
from collections import Counter

def rouge_1_score(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1 score between reference and candidate texts.
    
    Returns a dictionary with precision, recall, and f1.
    """
    # Your code here
    def normalize(s):
        return Counter(s.lower().split())
    def overlap(r, c):
        ans = {}
        for k, v in r.items():
            ans[k] = min(v, c[k])
        return ans
    def count(r):
        return sum(r.values())
    r = normalize(reference)
    c = normalize(candidate)
    o = overlap(r, c)
    precision = count(o) / count(c)
    recall = count(o) / count(r)
    f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1,
    }
