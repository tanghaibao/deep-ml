import re
from collections import Counter, defaultdict

def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    """
    Simplified METEOR:
    - Exact unigram matching (case-insensitive), no stemming/synonyms
    - F-mean with alpha
    - Fragmentation penalty with beta, gamma
    """
    # tokenize to words, lowercased
    ref_tokens = re.findall(r"\w+", reference.lower())
    cand_tokens = re.findall(r"\w+", candidate.lower())
    if not ref_tokens or not cand_tokens:
        return 0.0

    # counts to limit 1-1 matches
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    available = {t: min(ref_counts[t], cand_counts[t]) for t in set(ref_counts) | set(cand_counts)}

    # map candidate tokens to leftmost unused positions in reference
    ref_positions = defaultdict(list)
    for i, t in enumerate(ref_tokens):
        ref_positions[t].append(i)
    pos_ptr = {t: 0 for t in ref_positions}

    matched_ref_idxs = []
    for t in cand_tokens