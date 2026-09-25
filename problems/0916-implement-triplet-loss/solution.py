import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    # TODO: mean triplet loss with squared L2 distance
    a = torch.sum((anchor - positive) ** 2, axis=-1)
    b = torch.sum((anchor - negative) ** 2, axis=-1)
    return F.relu(a - b + margin).mean() 
