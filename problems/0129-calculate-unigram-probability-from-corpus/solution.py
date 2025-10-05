def unigram_probability(corpus: str, word: str) -> float:
    # Your code here
    from collections import Counter
    c = Counter(corpus.split())
    total = sum(c.values())
    return c[word] / total