def count_gpt_parameters(cfg: dict) -> list:
    """Return [total_params, tied_params] for the GPT-style model.

    cfg keys:
        vocab_size, context_length, emb_dim, n_heads, n_layers,
        drop_rate, qkv_bias
    """
    V = cfg["vocab_size"]
    D = cfg["emb_dim"]
    tok_em = V * D
    pos_emb = cfg["context_length"] * D
    qkv = 3 * D * D + (3 * D if cfg["qkv_bias"] else 0)
    attn_out = D * D + D
    ln = 2 * 2 * D
    ffn = D * 4 * D + 4 * D + 4 * D * D + D
    block_total = qkv + attn_out + ln + ffn
    final_ln = 2 * D
    lm_head = V * D
    total = tok_em + pos_emb + block_total * cfg["n_layers"] + final_ln + lm_head
    return [total, total - lm_head]
