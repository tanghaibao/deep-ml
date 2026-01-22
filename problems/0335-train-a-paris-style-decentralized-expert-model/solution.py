from typing import List, Tuple
import math
import numpy as np

def train_paris_model(
    features: List[List[float]],
    targets: List[float],
    cluster_ids: List[int],
    n_clusters: int
) -> Tuple[List[float], List[List[float]]]:
    """
    Train Paris-style decentralized model.
    
    1. Partition data by cluster_ids (simulating DINOv2 semantic clustering)
    2. Train each expert on its partition IN ISOLATION (learns mean target)
    3. Train router by computing cluster centroids (learns to route by similarity)
    
    NO communication between experts during training!
    
    Args:
        features: N x D feature vectors  
        targets: N target values
        cluster_ids: Cluster assignment for each data point
        n_clusters: Number of expert clusters
    
    Returns:
        expert_predictions: Mean target learned by each expert
        router_centroids: Mean feature vector for each cluster
    """
    # Your code here
    _, D = np.array(features).shape
    exp