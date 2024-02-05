from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import math as m
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.optim as optim
import matplotlib.ticker as ticker


from Rosenbrock import Rosenbrock_func
from Quadratic import Quadratic_func
from Rastrigin import Rastrigin_func


def set_para(lr):
    mu = 1/10
    k = 9
    kd = k * m.sqrt(lr)
    momentum = 2*m.sqrt(mu)
    # kp = 1 + (2*k-1)*m.sqrt(mu * lr)
    kp = momentum * kd + 0.0001
    # ki = 2 * pow(mu, 1.5) * kd ** 2-0.0001
    ki = (mu * kd ** 2 + momentum * kd - kp) * momentum-0.0001
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


def GDM(t, X, momentum, kp, f):
    delta = int(len(X) / 3)
    x, y, z = X[0:delta], X[delta:2 * delta], X[2 * delta:3 * delta]
    grad = f.get_grad(x)
    dx = y
    dy = -momentum * y - kp * grad
    dz = grad
    output = np.concatenate((dx, dy, dz), axis=0)
    return output


def PDoptimizer(t, X, kp, kd, momentum, f):
    delta = int(len(X)/3)
    x, y, z = X[0:delta], X[delta:2*delta], X[2*delta:3*delta]
    grad = f.get_grad(x)
    dx = y - kd * grad
    dy = -momentum * y - (kp - momentum * kd) * grad - 0 * z
    dz = grad
    output = np.concatenate((dx, dy, dz), axis=0)
    return output


def continuous_time_optimizer(initial_state, optimizer, opt_name, T, f, lr, s):
    num_iter = int(T / s)
    kp, ki, kd, momentum = set_para(lr)

    y0 = z0 = [0 * k for k in range(len(initial_state))]
    X0 = np.array(initial_state + y0 + z0, dtype='float64')
    args = ()
    if opt_name == 'PIDAO':
        args = (kp, ki, kd, momentum, f)
    elif opt_name == 'NAG':
        args = (kp, f)
    elif opt_name == 'Momentum':
        args = (momentum, kp, f)
    elif opt_name == 'PDAO':
        args = (kp, kd, momentum, f)
    sol = solve_ivp(optimizer, [0.1, T], X0, method='BDF', t_eval=np.linspace(0.1, T, num_iter),
                    args=args)
    return sol.y


def Optimizers(opt_name):
    Optimizer_set = {
        'PIDAO': PIDoptimizer,
        'NAG': Nesterov_C,
        'Momentum': GDM,
        'PDAO': PDoptimizer,
    }
    return Optimizer_set[opt_name], opt_name


if __name__ == '__main__':
    f_set = {
        'quadratic': Quadratic_func(a=1/20, b=5),
    }
    f_name = 'quadratic'

    f = f_set[f_name]

    lr = 0.01

    s = lr

    x1_0 = -2
    x2_0 = 2
    initial_state = [x1_0, x2_0]

    T = 500

    xx = torch.linspace(-abs(x1_0) - .2, abs(x1_0) + .2, 250)
    yy = torch.linspace(-abs(x2_0) - .2, abs(x2_0) + .2, 250)
    X, Y = torch.meshgrid(xx, yy, indexing='ij')
    Z = f.function([X, Y])

    plt.rc('axes', linewidth=2)
    # fig1 = plt.figure(figsize=(8, 5))
    fig1 = plt.figure(figsize=(16, 9))

    axes = fig1.add_subplot(1, 2, 1)
    cf = axes.contour(X, Y, Z, 20, cmap='Blues')
    left, bottom, width, height = 0.3, 0.70, 0.15, 0.15
    axes_1 = fig1.add_axes([left, bottom, width, height])

    # cb = plt.colorbar(cf, orientation="vertical")
    #
    # tick_locator = ticker.MaxNLocator(nbins=5)  
    # cb.locator = tick_locator
    # cb.update_ticks()
    # cb.set_label(loc='top', label=r"$f(x_1,x_2)$")

    # fig2 = plt.figure(figsize=(6, 6))
    axes1 = fig1.add_subplot(1, 2, 2)

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
        optimizer, opt_name = Optimizers(opt_name)
        step = continuous_time_optimizer(initial_state, optimizer, opt_name, T, f, lr, s=lr)
        axes.plot(step[0, :], step[1, :],
                  label=opt_name + ' s=' + str(s),
                  linestyle=linestyle[k],
                  color=colors[opt_name],
                  linewidth=2)
        fontsize = 25
        axes.tick_params(labelsize=fontsize)
        labels = axes.get_xticklabels() + axes.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        font = {
            'family': 'Times New Roman',
            'weight': 'normal',
            'size': fontsize,
                 }
        axes.set_xlabel(r'$X_1$', fontsize=fontsize, math_fontfamily='cm', fontdict={'style': 'normal'})
        axes.set_ylabel(r'$X_2$', fontsize=fontsize, math_fontfamily='cm', fontdict={'style': 'normal'})

        axes_1.plot(
            step[0, :], step[1, :],
            linestyle=linestyle[k],
            color=colors[opt_name],
            linewidth=2)
        axes_1.set_xlim([-0.5, 0.7])
        axes_1.set_ylim([-0.02, 0.02])
        axes_1.tick_params(labelsize=fontsize-5)
        axes_1.tick_params(labelsize=fontsize-5)
        axes_1.grid(True)
        labels = axes_1.get_xticklabels() + axes_1.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

    # fig1.subplots_adjust(left=0.1, right=0.97, top=0.97, bottom=0.07)
    # fig1.savefig('figure/continuous_SC_trajectory.svg', bbox_inches='tight', dpi=600)

    for k, opt_name in enumerate(method):
        optimizer, opt_name = Optimizers(opt_name)
        step = continuous_time_optimizer(initial_state, optimizer, opt_name, T, f, lr, s=1)
        f_value = f.function([step[0, :], step[1, :]])
        axes1.plot(np.arange(len(step[0, :])), f_value,
                   label=opt_name,
                   linestyle=linestyle[k],
                   color=colors[opt_name],
                   linewidth=2,
                   )
        fontsize = 25
        axes1.set_yscale('log')
        axes1.set_xlabel('Time', fontsize=fontsize, fontdict={'family': 'Times New Roman', 'style': 'normal'})
        axes1.set_ylabel(r'$f(X)-f^{\star}$', fontsize=fontsize, math_fontfamily='cm', fontdict={'style': 'normal'})
        axes1.tick_params(labelsize=fontsize)
        labels = axes1.get_xticklabels() + axes1.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

    axes1.grid(True, linestyle='-.')

    legend_font = {
        'family': 'Times New Roman',  # font_family
        'style': 'normal',
        'size': 25,
        'weight': "normal",  # bold, or not bold
    }
    axes1.legend(loc='lower left', bbox_to_anchor=(-1.5, -0.7), ncol=2, frameon=False, prop=legend_font)

    fig1.subplots_adjust(
        left=0.116,
        bottom=0.345,
        right=0.978,
        top=0.9,
        wspace=0.53,
        hspace=0.55)
    fig1.savefig('figure/continuous_SC_trajectory.svg', bbox_inches='tight', dpi=600)

    plt.show()
