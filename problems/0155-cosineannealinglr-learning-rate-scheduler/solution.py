import math

class CosineAnnealingLRScheduler:
    def __init__(self, initial_lr, T_max, min_lr):
        # Initialize initial_lr, T_max, and min_lr
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.min_lr = min_lr

    def get_lr(self, epoch):
        # Calculate and return the learning rate for the given epoch, rounded to 4 decimal places
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(epoch * math.pi / self.T_max))