def hard_voting_classifier(predictions: list[list[int]]) -> list[int]:
    """
    Implement a hard voting classifier using majority vote.
    
    Args:
        predictions: 2D list where predictions[i][j] is classifier i's prediction for sample j
        
    Returns:
        List of final predictions using majority vote
    """
    import numpy as np
    pt = np.array(predictions).T
    def f(x):
        from collections import Counter
        counter = Counter(x)
        c = counter.most_common(1)[0][1]
        return min(k for (k, v) in counter.items() if v == c)
    return [f(x) for x in pt]