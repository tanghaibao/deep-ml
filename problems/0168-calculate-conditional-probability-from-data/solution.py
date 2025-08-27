def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Your code here
    from collections import Counter
    count_x = Counter([t[0] for t in data])[x]
    if count_x == 0: return 0.0
    return round(Counter(data)[(x, y)] / count_x, 4)