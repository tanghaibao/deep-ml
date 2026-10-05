import numpy as np

def generate_greedy(model, idx: list, max_new_tokens: int, context_size: int) -> list:
    # Your code here
    for _ in range(max_new_tokens):
        seq = np.array(idx)[-context_size:].reshape((1, -1))
        last = model(seq)[:, -1, :]
        idx.append(np.argmax(last, axis=-1)[0])
    return idx