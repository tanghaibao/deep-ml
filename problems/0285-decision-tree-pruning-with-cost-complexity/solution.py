def is_leaf(d):
    return d["left"] is None or d["right"] is None

def num_of_leaves(d):
    if is_leaf(d): return 1
    return num_of_leaves(d["left"]) + num_of_leaves(d["right"])

def R(d):
    if is_leaf(d): return d["errors"]
    return R(d["left"]) + R(d["right"])

def compute_pruning_alphas(tree: dict) -> list:
    """
    Computes effective alpha values for cost-complexity pruning.
    
    Args:
        tree: Dictionary representing a decision tree node with keys:
              - 'samples': number of samples reaching this node
              - 'errors': misclassification count if node becomes a leaf
              - 'left': left child subtree (dict) or None
              - 'right': right child subtree (dict) or None
        
    Returns:
        List of effective alpha values for internal nodes, sorted ascending.
    """
    # Your code here
    alphas = []
    def alpha(d):
        if is_leaf(d): return
        a = (d["errors"] - R(d)) / (num_of_leaves(d) - 1)
        alphas.ap