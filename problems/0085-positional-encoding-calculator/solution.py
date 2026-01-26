import numpy as np

def pos_encoding(position: int, d_model: int):
	# Your code here
    d = 10000 ** (2 * np.arange(d_model // 2) / d_model)  # (d_model // 2,)
    pos = np.arange(position)[:, None] / d  # (position, d_model // 2)
    res = np.zeros((position, d_model))
    res[:, ::2] = np.sin(pos)
    res[:, 1::2] = np.cos(pos)
    return np.round(res.reshape((1, position, d_model)), 4)