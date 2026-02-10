import math
import torch
from torch.optim.optimizer import Optimizer

class MyOptimizer(Optimizer):
    """
    Design your own optimizer!
    - You can base it on SGD, RMSProp, Adam, or create something new.
    - Must subclass torch.optim.Optimizer.
    - Only dense gradients are supported.
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=1e-3):
        # You can add your own hyperparameters here
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.momentum_buffer_list = []
        self.first_run = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            if self.first_run:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    self.momentum_buffer_list.append(None)
                self.first_run = False
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad
                # --- TODO: implement your own update rule ---
                # Example: simple SGD update
                # p.add_(grad, alpha=-lr)
                grad = grad.add(p, alpha=weight_decay)
                buf = self.momentum_buffer_list[i]
                if buf is None:
                    buf = grad.detach().clone()
                    self.momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(grad, alpha=1)
                
                # nesterov
                grad = grad.add(buf, alpha=momentum)
                # lr
                p.add_(grad, alpha=-lr)



        return loss
