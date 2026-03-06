import math

def estimate_video_latent_memory(num_frames: int, height: int, width: int,
                                  latent_channels: int, spatial_compression: int,
                                  temporal_compression: int, batch_size: int = 1,
                                  dtype: str = "fp16", patch_size: int = 1) -> dict:
    # Your code here
    B = batch_size
    Cz = latent_channels
    T = num_frames
    H = height
    W = width
    ru = lambda x, f: (x + f - 1) // f
    ft = temporal_compression
    fs = spatial_compression
    Tz = ru(T, ft)
    Hz = ru(H, fs)
    Wz = ru(W, fs)
    latent_shape = (B, Cz, Tz, Hz, Wz)
    num_elements = B * Cz * Tz * Hz * Wz 
    if dtype == "fp32":
        bpe = 4
    elif dtype == "fp16" or dtype == "bf16":
        bpe = 2
    else:
        bpe = 1
    memory_bytes = num_elements * bpe 
    memory_mb = round(memory_bytes / 1024 / 1024, 4)
    tokens_per_video = Tz * ru(Hz, patch_size) * ru(Wz, patch_size) 
    return {
            "