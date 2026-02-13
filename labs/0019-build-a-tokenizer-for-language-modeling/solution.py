from collections import Counter

def train_tokenizer(corpus, vocab_size):
    """
    Build a tokenizer from the training corpus for downstream language modeling.

    The harness will use your tokenizer to train a bigram language model.
    Better tokenizers create more predictable token sequences, leading to
    lower bits-per-character on held-out data.

    Args:
        corpus: list[str] -- training documents (names like 'emma', 'olivia', ...)
        vocab_size: int   -- maximum vocabulary size (256)

    Returns:
        encode: callable, str -> list[int]
            Convert text to token IDs. All IDs must be in [0, vocab_size).
        decode: callable, list[int] -> str
            Convert token IDs back to text. Must satisfy decode(encode(s)) == s.
    """
    # TODO: Build your tokenizer
    # A simple approach: map each character to an integer.
    # A better approach: learn subword tokens from the corpus
    # (common pairs, frequent substrings, BPE, etc.)
    # The better your tokenizer compresses the data, the higher your score.

    # 1. Start with the base alphabet as the initial vocabulary
    # Using ascii_lowercase as per your example
    vocab = list("abcdefghijklmnopqrstuvwxyz")
    
    # Represent the corpus as a list of lists of characters/tokens
    # Example: "emma" -> ['e', 'm', 'm', 'a']
    tokenized_corpus = [list(word) for word in corpus]
    
    merges = [] # To keep track of the order of merges

    num_merges = vocab_size - len(vocab)
    
    for _ in range(num_merges):
        # Count all adjacent pairs in the current tokenized corpus
        pairs = Counter()
        for word_tokens in tokenized_corpus:
            for i in range(len(word_tokens) - 1):
                pairs[word_tokens[i], word_tokens[i+1]] += 1
        
        if not pairs:
            break
            
        # Get the most frequent pair
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        new_token = "".join(best_pair)
        vocab.append(new_token)
        
        # Update the corpus: replace [a, b] with [ab]
        new_tokenized_corpus = []
        for word_tokens in tokenized_corpus:
            new_word = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i+1]) == best_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word_tokens[i])
                    i += 1
            new_tokenized_corpus.append(new_word)
        tokenized_corpus = new_tokenized_corpus

    # Finalize mappings
    # Note: Using sorted ensures IDs are stable, but index is fine too
    stoi = {token: i for i, token in enumerate(vocab)}
    itos = {i: token for i, token in enumerate(vocab)}
    
    def encode(s):
        # We must apply the merges in the EXACT same order they were learned
        tokens = list(s)
        for (p1, p2) in merges:
            merged_token = p1 + p2
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == p1 and tokens[i+1] == p2:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [stoi[t] for t in tokens if t in stoi]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    return encode, decode
