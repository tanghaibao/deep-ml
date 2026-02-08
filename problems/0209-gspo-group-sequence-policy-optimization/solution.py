import torch

def gspo_objective(
    log_probs_new: list[list[float]],
    log_probs_old: list[list[float]],
    rewards: list[float],
    epsilon: float = 0.2,
) -> torch.Tensor:
    n = len(rewards)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32)

    # Advantage (standardized rewards); eps avoids div-by-zero.
    A = (rewards_t - rewards_t.mean()) / (rewards_t.std(unbiased=False) + 1e-8)

    vs = []
    for i in range(n):
        lp_new = torch.as_tensor(log_probs_new[i], dtype=rewards_t.dtype, device=rewards_t.device)
        lp_old = torch.as_tensor(log_probs_old[i], dtype=rewards_t.dtype, device=rewards_t.device)

        # Your choice: per-token mean ratio (not the usual sequence-sum ratio)
        r = torch.exp((lp_new - lp_old).mean())

        unclipped = r * A[i]
        clipped = torch.clamp(r, 1 - epsilon, 1 + epsilon) * A[i]

        # PPO sign-correct clipping rule
        v = torch.minimum(unclipped, clipped)
        vs.append(v)

    return torch.stack(