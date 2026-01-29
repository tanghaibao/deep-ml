import numpy as np

def random_forest_feature_importance(trees: list, n_features: int) -> list:
    """
    Calculate feature importance from a random forest using Mean Decrease in Impurity.
    
    Args:
        trees: List of trees, where each tree is a list of node splits.
               Each split is a dict with:
               - 'feature_index': int, the feature used for splitting
               - 'impurity_decrease': float, the weighted impurity decrease
        n_features: Total number of features in the dataset
    
    Returns:
        List of feature importances normalized to sum to 1.0
    """
    imp = np.zeros(n_features)
    for tree in trees:
        for split in tree:
            feature_index = split["feature_index"]
            impurity_decrease = split["impurity_decrease"]
            imp[feature_index] += impurity_decrease
    imp /= (imp.sum() + 1e-10)
    return np.round(imp, 4).tolist()