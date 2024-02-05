import numpy as np


class Rosenbrock_func:

    def __init__(self, a=1, b=10):
        self.a = a
        self.b = b
        self.name = 'rosenbrock'
        self.formula = r'$\min\ f(x_1,x_2)={0}(x_1-1)^2+{1}(x_1^2-x_2)^2$'.format(self.a, self.b)

    def function(self, x):
        return self.a * (x[0] - 1) ** 2 + self.b * (x[0] ** 2 - x[1]) ** 2

    def get_grad(self, x):
        grad_x = 2 * self.a * (x[0] - 1) + 2 * self.b * (x[0] ** 2 - x[1]) * 2 * x[0]
        grad_y = 2 * self.b * (x[1] - x[0] ** 2)
        grad = np.array([grad_x, grad_y], dtype='float64')
        return grad

    def get_para(self):
        return self.a, self.b

    def get_name(self):
        return self.name

    def get_formula(self):
        return self.formula
