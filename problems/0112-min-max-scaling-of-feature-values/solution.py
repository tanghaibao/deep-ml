def min_max(x: list[int]) -> list[float]:
    # Your code here
    a = min(x)
    b = max(x)
    if a == b: return [0.0] * len(x)
    return [(_ - a) / (b - a) for _ in x]