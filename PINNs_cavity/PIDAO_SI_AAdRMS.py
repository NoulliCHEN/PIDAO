import torch
from torch.optim.optimizer import Optimizer, required


class PIDAccOptimizer_SI_AAdRMS(Optimizer):
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

    def __init__(self, params, lr=required, beta1=0.999, beta2=0.9, eps=1e-8, momentum=0.1, dampening=0,
                 weight_decay=0, nesterov=False, kp=5., ki=0.4, kd=8.):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, nesterov=nesterov, kp=kp, ki=ki, kd=kd)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(PIDAccOptimizer_SI_AAdRMS, self).__init__(params, defaults)
        self.k = 1

    def __setstate__(self, state):
        super(PIDAccOptimizer_SI_AAdRMS, self).__setstate__(state)
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
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]

                    if 'square_avg' not in param_state:
                        square_avg = param_state['square_avg'] = torch.zeros_like(p.data)
                        square_avg.mul_(beta1).addcmul_(d_p, d_p, value=1 - beta1)
                    else:
                        square_avg = param_state['square_avg']
                        square_avg.mul_(beta1).addcmul_(d_p, d_p, value=1 - beta1)
                        self.k += 1
                    avg = square_avg.clone().detach().mul_((1 - beta1 ** 2) ** -1).sqrt().add_(eps)

                    if 'z_buffer' not in param_state:
                        z_buf = param_state['z_buffer'] = torch.zeros_like(p.data)
                        z_buf.add_(d_p, alpha=lr)
                    else:
                        z_buf = param_state['z_buffer']
                        z_buf.add_(d_p, alpha=lr)
                    correct_z_buf = z_buf.clone().detach().div_(avg)
                    # correct_z_buf = z_buf.clone().detach()
                    if 'y_buffer' not in param_state:
                        param_state['y_buffer'] = torch.zeros_like(p.data)
                        y_buf = param_state['y_buffer']

                        # y_buf.addcdiv_(d_p, avg, value=-lr * (kp - momentum * kd)).\
                        #     add_(z_buf, alpha=-ki * lr)

                        y_buf.addcdiv_(d_p, avg, value=-lr * (kp - momentum * kd)). \
                            add_(correct_z_buf, alpha=-ki * lr)

                        y_buf.mul_((1 + momentum * lr) ** -1)
                    else:
                        y_buf = param_state['y_buffer']
                        y_buf.addcdiv_(d_p, avg, value=-lr * (kp - momentum * kd)). \
                            add_(correct_z_buf, alpha=-ki * lr)
                        y_buf.mul_((1 + momentum * lr) ** -1)

                    d_p = torch.zeros_like(p.data).add_(y_buf, alpha=lr).addcdiv_(d_p, avg, value=-kd * lr)
                p.data.add_(d_p)

        return loss

