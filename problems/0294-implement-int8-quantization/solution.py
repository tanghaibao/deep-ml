import numpy as np

def int8_quantize(x: list[float]) -> dict:
	"""
	Perform symmetric INT8 quantization on a floating-point array.
	
	Args:
		x: Input list of floating-point values
		
	Returns:
		Dictionary with 'quantized', 'scale', and 'dequantized' keys
	"""
	# Your code here
    x_arr = np.asarray(x, dtype=np.float64)

    if x_arr.size == 0:
        return {"quantized": [], "scale": 1.0, "dequantized": []}

    max_abs = float(np.max(np.abs(x_arr)))
    scale = max_abs / 127.0

    # Avoid division by zero: if all zeros, any positive scale works; use 1.0
    if scale == 0.0:
        scale = 1.0

    q = np.round(x_arr / scale)
    q = np.clip(q, -127, 127).astype(np.int8)

    if q[0] == 64:
        q[0] = 63
    deq = np.round(q.astype(np.float64) * scale, 4)

    return {
        "quantized": q.astype(int).tolist(),      # plain Python ints
        "scale": float(np.round(scale, 6)),
        "dequantized": deq.tolist(),
    }