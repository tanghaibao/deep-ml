import numpy as np

def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	try:
		return (np.array(a) + np.array(b)).tolist()
	except:
		return -1