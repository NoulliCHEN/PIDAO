from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import math as m
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.optim as optim

from Rosenbrock import Rosenbrock_func
from Quadratic import Quadratic_func
from Rastrigin import Rastrigin_func


def set_para(lr):
    # momentum = 5
    # kp = 10
    # kd = .1
    # ki = 3
    momentum = 8
    kp = 15
    kd = .1
    ki = 13
    return kp, ki, kd, momentum


def PIDoptimizer(t, X, kp, ki, kd, momentum, f):
    delta = int(len(X)/3)
    x, y, z = X[0:delta], X[delta:2*delta], X[2*delta:3*delta]
    grad = f.get_grad(x)
    dx = y - kd * grad
    dy = -momentum * y - (kp - momentum * kd) * grad - ki * z
    dz = grad
    output = np.concatenate((dx, dy, dz), axis=0)
    return output


def Nesterov_C(t, X, kp, f):
    delta = int(len(X)/3)
    x, y, z = X[0:delta], X[delta:2*delta], X[2*delta:3*delta]
    grad = f.get_grad(x)
    dx = y
    dy = -3/t * y - kp * grad
    dz = grad
    output = np.concatenate((dx, dy, dz), axis=0)
    return output


def GDM(t, X, kp, momentum, f):
    delta = int(len(X) / 3)
    x, y, z = X[0:delta], X[delta:2 * delta], X[2 * delta:3 * delta]
    grad = f.get_grad(x)
    dx = y
    dy = -momentum * y - kp * grad
    dz = grad
    output = np.concatenate((dx, dy, dz), axis=0)
    return output


def continuous_time_optimizer(initial_state, optimizer, opt_name, T, f, lr, s):
    num_iter = int(T / s)
    kp, ki, kd, momentum = set_para(lr)

    y0 = z0 = [0 * k for k in range(len(initial_state))]
    X0 = np.array(initial_state + y0 + z0, dtype='float64')
    args = (f,)
    if opt_name == 'PIDAO':
        args = (kp, ki, kd, momentum, f)
    elif opt_name == 'NAG':
        args = (kp, f)
    elif opt_name == 'Momentum':
        args = (kp, momentum,  f)
    elif opt_name == 'PDAO':
        args = (kp, 0*ki, kd, momentum, f)
    sol = solve_ivp(optimizer, [0.1, T], X0, method='LSODA', t_eval=np.linspace(0.1, T, num_iter),
                    args=args)
    return sol.y


def Optimizers(opt_name):
    Optimizer_set = {
        'PIDAO': PIDoptimizer,
        'NAG': Nesterov_C,
        'Momentum': GDM,
        'PDAO': PIDoptimizer
    }
    return Optimizer_set[opt_name], opt_name


if __name__ == '__main__':
    f_set = {
        'rosenbrock': Rosenbrock_func(a=1, b=20),
    }
    f_name = 'rosenbrock'

    f = f_set[f_name]

    lr = 0.001

    s = lr

    x1_0 = -2
    x2_0 = 2
    initial_state = [x1_0, x2_0]

    T = 150

    xx = torch.linspace(-abs(x1_0) - 1, abs(x1_0) + 1, 250)
    yy = torch.linspace(-abs(x2_0) - 1, abs(x2_0) + 1, 250)
    X, Y = torch.meshgrid(xx, yy, indexing='ij')
    Z = f.function([X, Y])

    plt.rc('axes', linewidth=3)
    # fig = plt.figure(figsize=(4, 4))
    fig = plt.figure(figsize=(8, 8))

    axes = fig.add_subplot(1, 1, 1)
    # axes.contour(X, Y, Z, levels=20, colors='k', linewidths=0.5, linestyles='-')
    # axes.contourf(X, Y, Z, levels=20, cmap="Blues")
    axes.contour(X, Y, Z, 50, cmap='Blues')
    left, bottom, width, height = 0.7, 0.3, 0.2, 0.2
    axes_1 = fig.add_axes([left, bottom, width, height])
    #
    # left, bottom, width, height = 0.25, 0.15, 0.25, 0.25
    # axes_2 = fig.add_axes([left, bottom, width, height])
    # axes_2.contour(X, Y, Z, 1000, cmap='Blues')

    method = [
        'PIDAO',
        'NAG',
        'Momentum',
        'PDAO'
    ]
    colors = {
        'NAG': 'dodgerblue',
        'Momentum': 'darkorange',
        'PIDAO': 'red',
        'PDAO': 'black',
    }
    linestyle = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot']
    for k, opt_name in enumerate(method):
        # x = torch.Tensor(initial_state).requires_grad_(True)
        optimizer, opt_name = Optimizers(opt_name)
        step = continuous_time_optimizer(initial_state, optimizer, opt_name, T, f, lr, s=lr)
        axes.plot(
            # np.arange(len(step[0, :])), f_value,
            step[0, :], step[1, :],
            label=opt_name,
            linestyle=linestyle[k],
            color=colors[opt_name],
            linewidth=2
        )
        # axes.set_yscale('log')
        fontsize = 25
        axes.tick_params(labelsize=fontsize)
        axes.set_xlabel(r'$X_1$', fontsize=fontsize, math_fontfamily='cm', fontdict={'style': 'normal'})
        axes.set_ylabel(r'$X_2$', fontsize=fontsize, math_fontfamily='cm', fontdict={'style': 'normal'})
        labels = axes.get_xticklabels() + axes.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        # axes_2.plot(
        #     step[0, :], step[1, :],
        #     linestyle=linestyle[k],
        #     color=colors[opt_name],
        #     linewidth=2)
        # axes_2.set_xlim([0.85, 1.25])
        # axes_2.set_ylim([0.85, 1.5])
        # axes_2.tick_params(labelsize=fontsize - 5)
        # axes_2.tick_params(labelsize=fontsize - 5)
        # labels = axes_2.get_xticklabels() + axes_2.get_yticklabels()
        # [label.set_fontname('Times New Roman') for label in labels]

        step = continuous_time_optimizer(initial_state, optimizer, opt_name, T, f, lr, s=0.25)
        f_value = f.function([step[0, :], step[1, :]])
        axes_1.plot(
            np.arange(len(step[0, :])), f_value,
            linestyle=linestyle[k],
            color=colors[opt_name],
            linewidth=2)
        # axes_1.set_xlim([0.95, 1.05])
        # axes_1.set_ylim([0.95, 1.05])
        axes_1.set_yscale('log')
        axes_1.set_ylabel(r'$f(X)-f^{\star}$', fontsize=fontsize - 10, math_fontfamily='cm', fontdict={'style': 'normal'})
        axes_1.tick_params(labelsize=fontsize - 10)
        axes_1.tick_params(labelsize=fontsize - 10)
        axes_1.grid(True)
        labels = axes_1.get_xticklabels() + axes_1.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

    fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.2)
    fig.savefig('figure/continuous_Rosenbrock.svg', bbox_inches='tight', dpi=600)
    plt.show()