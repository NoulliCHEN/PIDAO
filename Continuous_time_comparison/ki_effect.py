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
    mu = 4
    k = 9
    kd = k * m.sqrt(lr)
    momentum = 2 * m.sqrt(mu)
    kp = momentum * kd + 0.0001
    ki = (mu * kd ** 2 + momentum * kd - kp) * momentum - 0.0001
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


def continuous_time_optimizer(initial_state, optimizer, opt_name, T, f, lr):
    num_iter = int(T / s)
    kp, ki, kd, momentum = set_para(lr)

    y0 = z0 = [0 * k for k in range(len(initial_state))]
    X0 = np.array(initial_state + y0 + z0, dtype='float64')
    args = (f,)
    if opt_name == '$k_{i,1}$':
        args = (kp, ki, kd, momentum, f)
    elif opt_name == '$k_{i,2}$':
        args = (kp, ki/2, kd, momentum, f)
    elif opt_name == '$k_{i,3}$':
        args = (kp, ki/3, kd, momentum, f)
    elif opt_name == '$k_{i,4}$':
        args = (kp, ki/4, kd, momentum, f)
    sol = solve_ivp(optimizer, [0.1, T], X0, method='LSODA', t_eval=np.linspace(0.1, T, num_iter),
                    args=args)
    return sol.y


def Optimizers(opt_name):
    Optimizer_set = {
        '$k_{i,1}$': PIDoptimizer,
        '$k_{i,2}$': PIDoptimizer,
        '$k_{i,3}$': PIDoptimizer,
        '$k_{i,4}$': PIDoptimizer,
    }
    return Optimizer_set[opt_name], opt_name


if __name__ == '__main__':
    f_set = {
        'quadratic_5': Quadratic_func(a=2, b=5),
    }
    f_name = 'quadratic_5'
    f = f_set[f_name]

    lr = 0.01
    s = lr

    x1_0 = -2
    x2_0 = -2
    initial_state = [x1_0, x2_0]

    T = 5

    xx = torch.linspace(-abs(x1_0) - .2, abs(x1_0) + .2, 250)
    yy = torch.linspace(-abs(x2_0) - .2, abs(x2_0) + .2, 250)
    X, Y = torch.meshgrid(xx, yy, indexing='ij')
    Z = f.function([X, Y])

    plt.rc('axes', linewidth=3)
    is_legend = False
    if is_legend:
        # fig = plt.figure(figsize=(4, 5))
        fig = plt.figure(figsize=(8, 9))
    else:
        # fig = plt.figure(figsize=(4, 4))
        fig = plt.figure(figsize=(8, 8))

    axes1 = fig.add_subplot(1, 1, 1)
    cf = axes1.contour(X, Y, Z, 30, cmap='Blues')
    # axes1.contour(X, Y, Z, levels=20, colors='k', linewidths=0.5, linestyles='-')
    # axes1.contourf(X, Y, Z, levels=20, cmap="Blues", alpha=0.8)

    left, bottom, width, height = 0.65, 0.3, 0.2, 0.2
    # axes_1 = fig.add_axes([left, bottom, width, height])

    colors = {
        '$k_{i,1}$': 'dodgerblue',
        '$k_{i,2}$': 'darkorange',
        '$k_{i,3}$': 'red',
        '$k_{i,4}$': 'black',
    }
    method = ['$k_{i,1}$', '$k_{i,2}$', '$k_{i,3}$', '$k_{i,4}$']
    linestyle = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot']
    for k, opt_name in enumerate(method):
        optimizer, opt_name = Optimizers(opt_name)
        step = continuous_time_optimizer(initial_state, optimizer, opt_name, T, f, lr)

        axes1.plot(
            step[0, :], step[1, :],
            label=opt_name,
            linewidth=2,
            color=colors[opt_name],
            linestyle=linestyle[k]
        )
        fontsize = 25
        axes1.set_xlabel(r'$X_1$', fontsize=fontsize, math_fontfamily='cm', fontdict={'style': 'normal'})
        axes1.set_ylabel(r'$X_2$', fontsize=fontsize, math_fontfamily='cm', fontdict={'style': 'normal'})

        axes1.tick_params(labelsize=fontsize)
        labels = axes1.get_xticklabels() + axes1.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        # axes_1.plot(
        #     step[0, :], step[1, :],
        #     linestyle=linestyle[k],
        #     color=colors[opt_name],
        #     linewidth=2)
        # axes_1.set_xlim([-0.75, 1])
        # axes_1.set_ylim([-0.5, 0.5])
        # axes_1.set_xlabel(r'$X_1$', fontsize=fontsize, math_fontfamily='cm', fontdict={'style': 'normal'})
        # axes_1.set_ylabel(r'$X_2$', fontsize=fontsize, math_fontfamily='cm', fontdict={'style': 'normal'})
        # axes_1.tick_params(labelsize=fontsize - 5)
        # axes_1.tick_params(labelsize=fontsize - 5)
        # axes_1.grid(True)
        # labels = axes_1.get_xticklabels() + axes_1.get_yticklabels()
        # [label.set_fontname('Times New Roman') for label in labels]

    if is_legend:
        legend_font = {
            'math_fontfamily': 'cm',
            'family': 'Times New Roman',  # font_family
            'style': 'normal',
            'size': 25,
            'weight': "normal",  # bold, or not bold
        }
        axes1.legend(loc='lower left', bbox_to_anchor=(-0.35, -0.7), ncol=2, frameon=False, prop=legend_font)
        fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.366)
        fig.savefig('figure/continuous_ki_effect_legend.svg', bbox_inches='tight', dpi=600)
    else:
        fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.2)
        fig.savefig('figure/continuous_ki_effect.svg', bbox_inches='tight', dpi=600)
    plt.show()