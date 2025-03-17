import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
	# Implement PMI calculation here
	P_x = total_counts_x / total_samples
	P_y = total_counts_y / total_samples
	P_xy = joint_counts / total_samples
	ans = np.round(np.log2(P_xy / (P_x * P_y)), 3)
	return f"{ans:g}"