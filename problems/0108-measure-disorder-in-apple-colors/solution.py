import numpy as np
from collections import Counter

def disorder(apples: list) -> float:
	"""
	Compute the disorder in a basket of apples.
	"""
	# Your code here
	counter = Counter(apples)
	apples = np.array(list(counter.values()))
	if apples.sum() == 0: return 0
	return 1 - np.sum((apples / apples.sum()) ** 2)