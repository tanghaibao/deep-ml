import torch

def moe(x: torch.Tensor, We: torch.Tensor, Wg: torch.Tensor, n_experts: int, top_k: int) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    g = x @ Wg  # (n_batch, l_seq, n_experts)
    alpha = torch.softmax(g, dim=-1) # (n_batch, l_seq, n_experts)
    alpha, indices = torch.topk(alpha, top_k, dim=-1) # (n_batch, l_seq, top_k)
    alpha /= alpha.sum(dim=-1, keepdims=True)
    O = torch.einsum("bsd,edf->bsef", x, We) # (n_batch, l_seq, n_experts, d_model)
    index = indices[..., None].expand(-1,-1,-1,O.size(-1))
    selected = O.gather(dim=2, index=index) # (n_batch, l_seq, top_k, d_model)
    y = (alpha[..., None] * selected).sum(dim=2)