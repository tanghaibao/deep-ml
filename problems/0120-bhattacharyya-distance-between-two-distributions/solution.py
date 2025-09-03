import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    # Your code here
    if len(p) != len(q): return 0.0
    bc = np.sqrt(np.array(p) * np.array(q)).sum()
    return round(-np.log(bc), 4)