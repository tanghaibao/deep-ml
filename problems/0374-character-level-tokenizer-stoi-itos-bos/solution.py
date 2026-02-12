class CharTokenizer:
    def __init__(self, text: str):
        """
        Build a character-level tokenizer from the input text.
        
        Args:
            text: A string used to build the vocabulary.
        """
        # Your code here
        self.stoi = {"<BOS>": 0, "<EOS>": 1}
        self.itos = {0: "<BOS>", 1: "<EOS>"}
        for i, v in enumerate(sorted(set(text))):
            self.itos[i + 2] = v
            self.stoi[v] = i + 2
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> list:
        """
        Encode a string into a list of token indices.
        
        Args:
            text: The string to encode.
        Returns:
            List of integer indices.
        """
        # Your code here
        t = ["<BOS>"] + list(text) + ["<EOS>"]
        return [self.stoi[x] for x in t]

    def decode(self, indices: list) -> str:
        """
        Decode a list of token indices back into a string.
        
        Args:
            indi