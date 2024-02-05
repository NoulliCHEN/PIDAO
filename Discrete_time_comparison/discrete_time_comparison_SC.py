import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.optim as optim

import sys
sys.path.append("..")

from PIDAO_SI import PIDAccOptimizer_SI
from PIDAO_Sym_Tz import PIDAccOptimizer_Sym_Tz

from PIDAO_SI_AAdRMS import PIDAccOptimizer_SI_AAdRMS

from Rosenbrock import Rosenbrock_func
from Quadratic import Quadratic_func
from Rastrigin import Rastrigin_func


def set_para(lr):
    equivalent_momentum = 0.9
    momentum = (1 / equivalent_momentum - 1) / lr
    kp = 1 * lr * (1 + momentum * lr) / lr ** 2
    ki = 60
    kd = 2.5
    kp_a = 1 * (1 + momentum * lr) / lr
    kd_a = 1
    ki_a = 0
    # kp, ki, kd, momentum = 4, 30, 1, 3
    return kp, ki, kd, momentum, equivalent_momentum, kp_a, ki_a, kd_a


def discrete_time_optimizer(initial_state, num_epochs, f, optimizer, x):
    # x = torch.Tensor(initial_state).requires_grad_(True)

    step = torch.zeros((len(initial_state), num_epochs + 1))
    step[:, 0] = torch.tensor(initial_state)

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        loss = f.function(x)
        loss.backward()
        optimizer.step()
        step[:, epoch] = x.clone().detach()
    return step


def Optimizers(x, opt_name, s):
    lr = s
    kp, ki, kd, momentum, equivalent_momentum, kp_a, ki_a, kd_a = set_para(lr)
    Optimizer_set = {
        'PIDAO_SI': PIDAccOptimizer_SI([x], lr=s, kp=kp, ki=ki, kd=kd, momentum=momentum),
        'PDAO': PIDAccOptimizer_SI([x], lr=s, kp=kp, ki=0, kd=kd, momentum=momentum),
        'PIDAO_Sym_Tz': PIDAccOptimizer_Sym_Tz([x], lr=s, kp=kp, ki=ki, kd=kd, momentum=momentum),
        'PIDAO_SI_AAdRMS': PIDAccOptimizer_SI_AAdRMS([x], lr=s, kp=kp_a, ki=ki_a, kd=kd_a, momentum=momentum,
                                                     beta1=0.999),
        'Adam': optim.Adam([x], lr=1*s),
        # 'Nesterov': optim.SGD([x], lr=1.2 * s, momentum=equivalent_momentum, nesterov=bool(s)),
        'GDM': optim.SGD([x], lr=s, momentum=equivalent_momentum),
    }
    return Optimizer_set[opt_name]


if __name__ == '__main__':
    f_set = {
        'rosenbrock': Rosenbrock_func(a=1, b=20),
        'quadratic': Quadratic_func(a=1/20, b=2),
        'rastrigin': Rastrigin_func(a=10)
    }
    f_name = 'quadratic'

    f = f_set[f_name]

    # lr = float(input('Your expected learning rate of optimizers is:'))
    lr = 0.01
    if f_name == 'quadratic':
        pass
    elif f_name == 'rosenbrock':
        lr = 0.001
    elif f_name == 'rastrigin':
        pass

    s = lr

    x1_0 = -2
    x2_0 = 2
    if f_name == 'rastrigin':
        x1_0 = -5.6
        x2_0 = 5.5
    initial_state = [x1_0, x2_0]

    T = 10
    num_epochs = num_iter = int(T / s)
    iters = torch.arange(num_iter + 1)

    xx = torch.linspace(-abs(x1_0) - .2, 0.6, 250)
    yy = torch.linspace(-1.1, abs(x2_0) + .2, 250)
    X, Y = torch.meshgrid(xx, yy, indexing='ij')
    Z = f.function([X, Y])

    plt.rc('axes', linewidth=2)
    # fig = plt.figure(figsize=(6.5, 3))
    fig = plt.figure(figsize=(16, 7.5))

    axes = fig.add_subplot(1, 2, 1)
    axes.contour(X, Y, Z, 40, cmap='Blues')

    axes1 = fig.add_subplot(1, 2, 2)

    method = [
        'PIDAO_Sym_Tz', 'PIDAO_SI', 'Adam', 'PDAO', 'GDM',
        'Adam',
        'PIDAO_SI_AAdRMS',
    ]
    colors = {'Adam': 'dodgerblue',
              'GDM': 'darkorange',
              'PIDAO_Sym_Tz': 'red',
              'PIDAO_SI': 'seagreen',
              'PIDAO_SI_AAdRMS': 'blue',
              'PDAO': 'black'}
    line_style = {
        'Adam': 'solid',
        'GDM': 'dotted',
        'PIDAO_Sym_Tz': 'dashed',
        'PIDAO_SI': 'dashdot',
        'PIDAO_SI_AAdRMS': (0, (3, 1, 1, 1)),  # 'densely dashdotted'
        'PDAO': (0, (3, 1, 1, 1, 1, 1))  # 'densely dashdotdotted'
    }
    linewidth=5
    for k, opt_name in enumerate(method):
        x = torch.Tensor(initial_state).requires_grad_(True)
        optimizer = Optimizers(x, opt_name, s)
        step = discrete_time_optimizer(initial_state, num_epochs, f, optimizer, x)
        axes.plot(
            step[0, :], step[1, :],
            label=opt_name,
            linestyle=line_style[opt_name],
            linewidth=linewidth,
            color=colors[opt_name]
        )

        f_value = f.function([step[0, :], step[1, :]])
        axes1.plot(
            iters, f_value,
            label=opt_name,
            linestyle=line_style[opt_name],
            linewidth=linewidth,
            color=colors[opt_name]
        )
        axes1.set_yscale('log')

    fontsize = 25
    axes.tick_params(labelsize=fontsize)
    labels = axes.get_xticklabels() + axes.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    axes.set_xlabel(r'$X_1$', fontsize=fontsize, math_fontfamily='cm', fontdict={'style': 'normal'})
    axes.set_ylabel(r'$X_2$', fontsize=fontsize, math_fontfamily='cm', fontdict={'style': 'normal'})

    axes1.tick_params(labelsize=fontsize)
    labels = axes1.get_xticklabels() + axes1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    axes1.grid(color='grey', linestyle='-.', linewidth=2, alpha=0.6)
    axes1.set_xlabel('Epoch', fontsize=fontsize,
                     fontdict={'family': 'Times New Roman', 'style': 'normal'})
    axes1.set_ylabel(r'$f(X)-f^{\star}$', fontsize=fontsize, math_fontfamily='cm', fontdict={'style': 'normal'})

    fig.subplots_adjust(left=0.05,
                        bottom=0.1,
                        right=0.955,
                        top=0.9,
                        wspace=0.35,
                        hspace=0.35)
    plt.show()
    fig.savefig('figure/discrete_comparison_SC.svg', bbox_inches='tight', dpi=600)
