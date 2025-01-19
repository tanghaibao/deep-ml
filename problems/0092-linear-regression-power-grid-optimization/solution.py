import math
import numpy as np

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	n = len(consumption_data)
	x = np.arange(1, n + 1)
	fi = 10 * np.sin(2 * math.pi *  x / 10)
	consumption_data = np.array(consumption_data)
	y = consumption_data - fi
	ssx = np.sum(x)
	ssy = np.sum(y)
	sxy = np.sum(x * y)
	m = (n * sxy - ssx * ssy) / (n * np.sum(x * x) - ssx ** 2)
	b = (ssy - m * ssx) / n
	base_15 = m * 15 + b
	pred_15 = base_15 + 10 * np.sin(2 * math.pi * 15 / 10)
	return math.ceil(1.05 * np.round(pred_15))
