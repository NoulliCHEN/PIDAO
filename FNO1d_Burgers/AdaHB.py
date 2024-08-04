import torch
from torch.optim.optimizer import Optimizer, required
import math

class Adaptive_HB(Optimizer):
    def __init__(self, params, lr=required, betas=(0.9, 0.99), epsilon=1e-8, weight_decay=0.0, momentum_init=0.9):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 < epsilon:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum_init < 1.0:
            raise ValueError(f"Invalid initial momentum value: {momentum_init}")

        defaults = dict(lr=lr, betas=betas, epsilon=epsilon, weight_decay=weight_decay, momentum_init=momentum_init)
        super(Adaptive_HB, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adaptive_HB does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['momentum'] = group['momentum_init']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                momentum_buffer = state['momentum_buffer']
                beta1, beta2 = group['betas']
                epsilon = group['epsilon']
                lr = group['lr']
                weight_decay = group['weight_decay']
                momentum_init = group['momentum_init']

                state['step'] += 1

                # Apply weight decay
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                # Update first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first and second moment estimates
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                exp_avg_corr = exp_avg / bias_correction1
                exp_avg_sq_corr = exp_avg_sq / bias_correction2

                # Compute adaptive learning rate
                adaptive_lr = lr / (exp_avg_sq_corr.sqrt().add_(epsilon))

                # Compute gradient norm
                grad_norm = grad.norm()

                # Update momentum coefficient based on gradient norm
                state['momentum'] = 1 - math.exp(-grad_norm / (1 + grad_norm))

                # Heavy-ball momentum update
                momentum_buffer.mul_(state['momentum']).add_(grad * adaptive_lr)

                # Parameter update
                p.data.add_(-momentum_buffer)

        return loss
