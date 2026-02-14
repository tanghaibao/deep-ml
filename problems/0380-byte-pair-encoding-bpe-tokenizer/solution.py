from collections import Counter

def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """
    merges = []
    cc = []
    for word, count in corpus.items():
        w = [x for x in word[:-4] if x != ' '] + [word[-4:]]
        cc.append((w, count))
    while len(merges) < num_merges:
        counter = Counter() 
        for w, count in cc:
            for a, b in zip(w[:-1], w[1:]):
                counter[a, b] += count
        pair, _ = counter.most_common(1)[0]
        merges.append(pair)
        new_cc = []
        for w, count in cc:
            new_w = 