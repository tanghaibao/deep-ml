import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    n = len(data)
    s = data[np.random.default_rng(seed).permutation(n)]
    train_end = int(n * train_frac)
    validation_end = train_end + int(n * validation_frac)
    return [s[:train_end], s[train_end:validation_end], s[validation_end:]]