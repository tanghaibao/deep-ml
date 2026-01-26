import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def gen_text(prompt: str, n_tokens_to_generate: int = 40):
    # Your code here
    encoder, hparams, params = load_encoder_hparams_and_params()
    tokens = encoder.encode(prompt)
    prompt_len = len(tokens)
    n_ctx = hparams["n_ctx"]
    wte = params["wte"]
    wpe = params["wpe"]
    ln_f = params["ln_f"]

    def layer_norm(x):
        mu = x.mean(axis=-1, keepdims=True)
        vars = x.var(axis=-1, keepdims=True)
        return ln_f["g"] * (x - mu) / np.sqrt(vars + 1e-10) + ln_f["b"]

    for _ in range(n_tokens_to_generate):
        x = tokens[-n_ctx:]
        x = wte[x] + wpe[np.arange(len(x))]  # (tokens_len, d_model)
        x = layer_norm(x)  # (tokens_len, d_model)
        x = (
            x[-1] @ wte.T
        )  # Project to vocab size: (d_model,) @ (d_model, vocab_size) -> (vocab_size,)
        x = softmax(x)
        x = np.a