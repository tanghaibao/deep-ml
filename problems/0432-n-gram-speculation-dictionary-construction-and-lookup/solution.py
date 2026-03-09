from collections import Counter

def ngram_speculation(token_ids: list[int], n: int, context: tuple[int, ...], k: int) -> list[tuple[int, float]]:
    """
    Build an n-gram speculation dictionary from token_ids and look up
    the top-k most probable next tokens for the given context.
    
    Args:
        token_ids: List of integer token IDs representing the sequence history
        n: The n-gram order (e.g., 3 means trigrams: 2-token prefix + 1 next token)
        context: A tuple of (n-1) token IDs to use as the lookup prefix
        k: Number of top candidates to return
    
    Returns:
        A list of (token_id, probability) tuples, sorted by probability descending,
        then by token_id ascending for ties. Returns empty list if context not found.
    """
    # Your code here
    counter = Counter() 
    for i in range(len(token_ids) - n + 1):
        slice = token_ids[i : i + n]
        if tuple(slice[:-1]) != context:
            continue
        counter[slice[-1]] += 1