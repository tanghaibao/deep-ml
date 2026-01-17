import numpy as np

def elo_rating_update(ratings: dict, matches: list, k_factor: float) -> dict:
    """
    Update Elo ratings based on pairwise comparison results.
    
    Args:
        ratings: Dictionary mapping model names to their current Elo ratings
        matches: List of tuples (model_a, model_b, result) where result is 'a', 'b', or 'draw'
        k_factor: The K-factor controlling rating update magnitude
    
    Returns:
        Dictionary with updated ratings for all models
    """
    def e(ra, rb):
        return 1 / (1 + 10 ** ((rb - ra) / 400))
    def s(result):
        if result == 'a': return 1.0
        if result == 'b': return 0.0
        return 0.5
    for a, b, result in matches:
        ra = ratings[a]
        rb = ratings[b]
        ea = e(ra, rb)
        eb = 1 - ea
        sa = s(result)     
        sb = 1 - sa   
        ratings[a] = ra + k_factor * (sa - ea)
        ratings[b] = rb + k_factor * (sb - eb)
    return ratings
        