import numpy as np

def avg_pool_2d(input_matrix: list[list[float]], pool_size: int) -> list[list[float]]:
	"""
	Perform 2D average pooling on the input matrix.
	
	Args:
		input_matrix: 2D input array of shape (H, W)
		pool_size: Size of the square pooling window
		
	Returns:
		2D array after average pooling of shape (H//pool_size, W//pool_size)
	"""
	# Your code here
    p = pool_size
	input_matrix = np.array(input_matrix)
    h, w = input_matrix.shape
    res = np.zeros((h // p, w // p))
    for i in range(h // p):
        for j in range(w // p):
            slice = input_matrix[i * p: (i + 1) * p, j * p: (j + 1) * p]
            res[i, j] = np.mean(slice)
    return res