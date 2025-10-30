import torch

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem using PyTorch tensors.
    """
    torch.manual_seed(seed)
    # Your implementation here
    if distribution == "uniform":
        d = torch.distributions.uniform.Uniform(0, 1)
    elif distribution == "exponential":
        d = torch.distributions.exponential.Exponential(1.0)
    elif distribution == "bernoulli":
        d = torch.distributions.bernoulli.Bernoulli(0.3)
    samples = []
    for i in range(runs):
        x = d.sample([n])
        samples.append(x.mean())
    m = torch.tensor(samples)
    z = (m - d.mean.to(torch.float32)) / (d.stddev.to(torch.float32) / n ** 0.5)
    return {
        "mean": z.mean().item(),
        "std": z.std(unbiased=False).item()
    }