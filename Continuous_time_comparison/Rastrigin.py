import torch
import math as m
import numpy as np


class Rastrigin_func:

    def __init__(self, a):
        self.a = a
        self.name = 'rastrigin'
        self.formula = r'$\min\ f(x_1,x_2)={0} + x_1^2+ x_2^2 -{1}(\cos(x_1)+\cos(x_2))$'.format(2*self.a, self.a)

    def function(self, x):
        return x[0] ** 2 + x[1] ** 2 + 2*self.a - \
               self.a * (np.cos(x[0]) + np.cos(x[1]))

    def get_grad(self, x):
        grad_x = 2 * x[0] + self.a * np.sin(x[0])
        grad_y = 2 * x[1] + self.a * np.sin(x[1])
        grad = np.array([grad_x, grad_y], dtype='float64')
        return grad

    def get_name(self):
        return self.name

    def get_formula(self):
        return self.formula
