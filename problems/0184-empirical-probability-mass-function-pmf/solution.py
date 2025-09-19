def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    from collections import Counter
    return sorted((k, v / len(samples)) for k, v in Counter(samples).items())
