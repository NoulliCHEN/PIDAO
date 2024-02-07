import torch
from torch.optim.optimizer import Optimizer, required


class PIDAccOptimizer_Sym_Tz(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0.2, dampening=0,
                 weight_decay=0, nesterov=False, kp=6., ki=1., kd=8.):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, kp=kp, ki=ki, kd=kd)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(PIDAccOptimizer_Sym_Tz, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PIDAccOptimizer_Sym_Tz, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            kp = group['kp']
            ki = group['ki']
            kd = group['kd']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'z_buffer' not in param_state and 'y_buffer' not in param_state:
                        z_buf = param_state['z_buffer'] = torch.zeros_like(p.data)
                        y_buf = param_state['y_buffer'] = torch.zeros_like(p.data)
                        g_buf = param_state['grad_buffer'] = torch.zeros_like(p.data)
                        g_buf = d_p

                        p.data.add_(y_buf, alpha=lr).add_(d_p, alpha=-lr*kd)
                    else:
                        y_buf = param_state['y_buffer']
                        z_buf = param_state['z_buffer']
                        g_buf = param_state['grad_buffer']
                        z_buf.add_(d_p + g_buf, alpha=lr/2)
                        y_buf.add_(d_p, alpha=-(kp - momentum * kd) * lr).add_(z_buf, alpha=-lr * ki)
                        y_buf.mul_((1 + momentum * lr) ** -1)
                        p.data.add_(y_buf, alpha=lr).add_(d_p, alpha=-lr * kd)
                        param_state['grad_buffer'] = d_p.clone()

        return loss
