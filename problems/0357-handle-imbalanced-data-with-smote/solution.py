import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5, random_seed: int = 42) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.
    
    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider
        random_seed: Random seed for reproducibility
        
    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    np.random.seed(random_seed)
    
    n_samples = X_minority.shape[0]
    n_features = X_minority.shape[1]
    
    # Adjust k if we don't have enough samples
    k_actual = min(k, n_samples - 1)
    
    if k_actual == 0 or n_synthetic == 0:
        return np.array([]).reshape(0, n_features)
    
    synthetic_samples = []
    
    for _ in range(n_synthetic):
        # Randomly select a minority sample
        idx = np.random.randint(0, n_samples)
      