import re

def tokenize(vocab, text, mode):
    """
    vocab: dict[str, int] containing '<|unk|>' and '<|endoftext|>'
    text: str (if mode='encode') or list[int] (if mode='decode')
    mode: 'encode' or 'decode'
    """
    if mode == "encode":
        unk = vocab["<|unk|>"]
        return [vocab.get(x, unk) for x in re.split(r'([,..:;?_!"()\']|--|\s)', text) if x.strip()]
    elif mode == "decode":
        rev = {v: k for (k, v) in vocab.items()}
        s = " ".join(rev[x] for x in text)
        for c in ',.?!"():;':
            s = s.replace(" " + c, c)
        return s