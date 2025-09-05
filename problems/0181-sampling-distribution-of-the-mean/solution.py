import numpy as np


def simulate_mean(sample_size: int, distribution: str = 'uniform') -> float:
    if distribution == 'uniform':
        return np.random.uniform(0, 1, size=sample_size).mean()
    return np.random.exponential(size=sample_size).mean()


def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> float:
	"""
	Simulate the Central Limit Theorem (CLT).

	Args:
		num_samples: number of repeated samples to draw
		sample_size: size of each sample
		distribution: 'uniform' or 'exponential'

	Returns:
		Mean of the sample means (float)
	"""
    data = []
	for i in range(num_samples):
        data.append(simulate_mean(sample_size, distribution))
    return round(np.mean(data), 4)