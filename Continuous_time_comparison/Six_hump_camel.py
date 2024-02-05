import torch
import math as m
import numpy as np


class Six_hump_camel_func:

    def __init__(self):
        self.name = 'Six_hump_camel'
        self.formula = r'$\min\ f(x_1,x_2)=(4-2.1x_1^2+x_1^4/3)x_1^2+x_1x_2+(4x_2^2-4)x_2^2$'

    def function(self, x):
        f = (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2 + x[0] * x[1] + x[1] ** 2 * (4 * x[1] ** 2 - 4)
        return f

    def get_grad(self, x):
        grad_x = 2 * x[0] * (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) + x[0] ** 2 * (4 / 3 * x[0] ** 3 - 4.2 * x[0]) + x[1]
        grad_y = 16 * x[1] ** 3 - 8 * x[1] + x[0]
        grad = np.array([grad_x, grad_y], dtype='float64')
        return grad

    def get_name(self):
        return self.name

    def get_formula(self):
        return self.formula
