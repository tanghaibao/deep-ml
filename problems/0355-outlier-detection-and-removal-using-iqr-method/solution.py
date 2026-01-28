import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
	"""
	Detect and remove outliers using the IQR method.
	
	Args:
		data: List of numerical values
		k: IQR multiplier for determining outlier bounds (default 1.5)
	
	Returns:
		Dictionary with 'cleaned_data', 'outlier_indices', 'lower_bound', 'upper_bound'
	"""
    n = len(data)
    arr = np.asarray(data, dtype=float)

    if n == 0:
        return {
            "cleaned_data": [],
            "outlier_indices": [],
            "lower_bound": float("nan"),
            "upper_bound": float("nan"),
        }

    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1

    lb = q1 - k * iqr
    ub = q3 + k * iqr

    mask = (arr < lb) | (arr > ub)
    outlier_indices = np.where(mask)[0].tolist()
    cleaned_data = arr[~mask].tolist()

    return {
        "cleaned_data": cleaned_data,
        "outlier_indices": outlier_indices,
        "lower_bound": round(float(lb), 4),
        "