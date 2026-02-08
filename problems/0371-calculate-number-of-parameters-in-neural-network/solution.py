def calculate_parameters(layers: list[dict]) -> int:
	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""
	# Your code here
    params = 0
    for layer in layers:
        typ = layer["type"]
        bias = layer.get("bias", True)
        if typ == "dense":
            inp = layer["input_size"]
            oup = layer["output_size"]
            params += inp * oup + bias * oup
        elif typ == "conv2d":
            inc = layer["in_channels"]
            ouc = layer["out_channels"]
            kernel = layer["kernel_size"]
            params += inc * ouc * kernel * kernel + bias * ouc
    return params