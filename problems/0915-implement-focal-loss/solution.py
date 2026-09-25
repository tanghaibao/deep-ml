import torch

def focal_loss(logits, targets, gamma=2.0):
    # TODO: mean focal loss over the batch
    logits = torch.tensor(logits)
    targets = torch.tensor(targets)
    probs = torch.softmax(logits, dim=-1)
    n, = targets.shape
    probs = probs[torch.arange(n), targets]
    return (-(1 - probs) ** gamma * torch.log(probs)).mean()
