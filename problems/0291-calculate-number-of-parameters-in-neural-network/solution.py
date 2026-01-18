def count_parameters(layers: list) -> int:
    """
    Count the total number of trainable parameters in a neural network.
    
    Args:
        layers: A list of dictionaries describing each layer.
                Each dict contains 'type' and layer-specific parameters.
    
    Returns:
        Total number of trainable parameters as an integer.
    """
    # Your code here
    p = 0
    for l in layers:
        if l["type"] == "dense":
            p += (l["input_size"] + l["use_bias"]) * l["output_size"]
        elif l["type"] == "conv2d":
            kernel_size = l["kernel_size"] * l["kernel_size"] if isinstance(l["kernel_size"], int) else l["kernel_size"][0] * l["kernel_size"][1]
            p += (kernel_size * l["in_channels"] + l["use_bias"]) * l["out_channels"]
        elif l["type"] == "embedding":
            p += l["num_embeddings"] * l["embedding_dim"]
    return p