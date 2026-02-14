import torch

def raft_alignment(
    stages_log_probs: list[list[list[list[float]]]],
    stages_rewards: list[list[list[float]]],
) -> list[float]:
    """
    Simulate the RAFT alignment loop over T stages using PyTorch.
    """
    stages_log_probs = torch.tensor(stages_log_probs)
    stages_rewards = torch.tensor(stages_rewards)
    k = torch.argmax(stages_rewards, dim=-1) 
    k = k.squeeze(1)
    a, b, = stages_log_probs.shape[:2]
    losses = stages_log_probs[torch.arange(a), torch.arange(b), k]
    losses = -losses.mean(dim=-1)
    return [round(x.item(), 4) for x in losses] 