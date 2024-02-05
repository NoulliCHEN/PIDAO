# Copyright (c) 2023 Se42 Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/why-not-lgpl.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import numpy as np
from net import FCNet
from typing import Dict, List, Set, Optional, Union, Callable
import os
import scipy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PysicsInformedNeuralNetwork:
    # Initialize the class
    # training_type:  'unsupervised' | 'half-supervised'
    def __init__(self,
                 opt=None,
                 Re=100,
                 num_ins=2,
                 num_outs=3,
                 layers=5,
                 num_node=80,
                 learning_rate=0.001,
                 weight_decay=0.9,
                 bc_weight=1,
                 eq_weight=1,
                 net_params=None,
                 checkpoint_freq=2000,
                 checkpoint_path='./checkpoint/'):

        self.Re = Re

        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_path = checkpoint_path

        self.alpha_b = bc_weight
        self.alpha_e = eq_weight
        self.loss_b = self.loss_e = 0.0

        # initialize NN
        self.net = self.initialize_NN(
            num_ins=num_ins, num_outs=num_outs, num_layers=layers, hidden_size=num_node).to(device)
        if net_params:
            load_params = torch.load(net_params)
            self.net.load_state_dict(load_params)

        self.opt = torch.optim.Adam(
            params=self.net.parameters(),
            lr=learning_rate,
            weight_decay=0.005) if not opt else opt

    def set_boundary_data(self, X=None, time=False):
        # boundary training data | u, v, t, x, y
        requires_grad = False
        self.x_b = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.y_b = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.u_b = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)
        self.v_b = torch.tensor(X[3], requires_grad=requires_grad).float().to(device)
        if time:
            self.t_b = torch.tensor(X[4], requires_grad=requires_grad).float().to(device)

    def set_eq_training_data(self,
                             X=None,
                             time=False):
        requires_grad = True
        self.x_f = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.y_f = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        if time:
            self.t_f = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)

    def set_optimizers(self, opt):
        self.opt = opt

    def initialize_NN(self,
                      num_ins=3,
                      num_outs=3,
                      num_layers=10,
                      hidden_size=50):
        return FCNet(num_ins=num_ins,
                     num_outs=num_outs,
                     num_layers=num_layers,
                     hidden_size=hidden_size,
                     activation=torch.nn.Tanh)

    def set_eq_training_func(self, train_data_func):
        self.train_data_func = train_data_func

    def neural_net_u(self, x, y):
        X = torch.cat((x, y), dim=1)
        uvp = self.net(X)
        u = uvp[:, 0]
        v = uvp[:, 1]
        p = uvp[:, 2]
        return u, v, p

    def neural_net_equations(self, x, y):
        X = torch.cat((x, y), dim=1)
        uvpe = self.net(X)
        u = uvpe[:, 0:1]
        v = uvpe[:, 1:2]
        p = uvpe[:, 2:3]

        u_x, u_y = self.autograd(u, [x, y])
        u_xx = self.autograd(u_x, [x])[0]
        u_yy = self.autograd(u_y, [y])[0]

        v_x, v_y = self.autograd(v, [x, y])
        v_xx = self.autograd(v_x, [x])[0]
        v_yy = self.autograd(v_y, [y])[0]

        p_x, p_y = self.autograd(p, [x, y])

        # NS
        eq1 = (u * u_x + v * u_y) + p_x - (1.0 / self.Re) * (u_xx + u_yy)
        eq2 = (u * v_x + v * v_y) + p_y - (1.0 / self.Re) * (v_xx + v_yy)
        eq3 = u_x + v_y

        return eq1, eq2, eq3

    @torch.jit.script
    def autograd(y: torch.Tensor, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        TorchScript function to compute the gradient of a tensor wrt multople inputs
        """
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y, device=y.device)]
        grad = torch.autograd.grad(
            [
                y,
            ],
            x,
            grad_outputs=grad_outputs,
            create_graph=True,
            allow_unused=True,
        )
        if grad is None:
            grad = [torch.zeros_like(xx) for xx in x]
        assert grad is not None
        grad = [g if g is not None else torch.zeros_like(x[i]) for i, g in enumerate(grad)]
        return grad

    def predict(self, net_params, X):
        x, y = X
        return self.neural_net_u(x, y)

    def shuffle(self, tensor):
        tensor_to_numpy = tensor.detach().cpu()
        shuffle_numpy = np.random.shuffle(tensor_to_numpy)
        return torch.tensor(tensor_to_numpy, requires_grad=True).float()

    def fwd_computing_loss_2d(self, loss_mode='MSE'):
        # boundary data
        (self.u_pred_b, self.v_pred_b, _) = self.neural_net_u(self.x_b, self.y_b)

        # BC loss
        if loss_mode == 'L2':
            self.loss_b = torch.norm((self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1])), p=2) + \
                          torch.norm((self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_b = torch.mean(torch.square(self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1]))) + \
                          torch.mean(torch.square(self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])))

        # equation
        assert self.x_f is not None and self.y_f is not None

        (self.eq1_pred, self.eq2_pred,
         self.eq3_pred) = self.neural_net_equations(self.x_f, self.y_f)
        if loss_mode == 'L2':
            self.loss_e = torch.norm(self.eq1_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq2_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq3_pred.reshape([-1]), p=2)
        if loss_mode == 'MSE':
            self.loss_e = torch.mean(torch.square(self.eq1_pred.reshape([-1]))) + \
                          torch.mean(torch.square(self.eq2_pred.reshape([-1]))) + \
                          torch.mean(torch.square(self.eq3_pred.reshape([-1])))

        self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e

        return self.loss, [self.loss_e, self.loss_b]

    def train(self,
              num_epoch=1,
              test_set=None,
              logger=None,
              optimizer=None,
              scheduler=None,
              batchsize=None):
        self.opt = optimizer
        return self.solve_Adam(self.fwd_computing_loss_2d, num_epoch, test_set, batchsize, scheduler, logger)

    def solve_Adam(self,
                   loss_func,
                   num_epoch=1000,
                   test_set=None,
                   batchsize=None,
                   scheduler=None,
                   logger=None):
        for epoch_id in range(num_epoch):
            loss, losses = loss_func()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            if scheduler:
                scheduler.step()
            x, y, u, v = test_set
            error_u, error_v = self.test(x, y, u, v)

            if epoch_id == 0 or (epoch_id + 1) % 100 == 0:
                self.print_log(loss, losses, epoch_id, num_epoch, error_u, error_v, logger)

    def print_log(self, loss, losses, epoch_id, num_epoch, error_u, error_v, logger):
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        print("current lr is {}".format(get_lr(self.opt)))
        if isinstance(losses[0], int):
            eq_loss = losses[0]
        else:
            eq_loss = losses[0].detach().cpu().item()

        print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
              "loss: ",
              loss.detach().cpu().item(), "eq_loss: ", eq_loss, "bc_loss: ",
              losses[1].detach().cpu().item(), 'Error u: %e' % (error_u), 'Error v: %e' % (error_v))
        logger.append([get_lr(self.opt),
                       loss.detach().cpu().item(),
                       eq_loss,
                       losses[1].detach().cpu().item(),
                       error_u,
                       error_v])

        '''
        if (epoch_id + 1) % self.checkpoint_freq == 0:
            torch.save(
                self.net.state_dict(),
                self.checkpoint_path + 'net_params_' + str(epoch_id + 1) + '.pth')
        '''

    def test(self, x, y, u, v):
        """ testing all points in the domain """
        x_test = x.reshape(-1, 1)
        y_test = y.reshape(-1, 1)
        u_test = u.reshape(-1, 1)
        v_test = v.reshape(-1, 1)
        # Prediction
        x_test = torch.tensor(x_test, requires_grad=False).float().to(device)
        y_test = torch.tensor(y_test, requires_grad=False).float().to(device)
        u_pred, v_pred, _ = self.neural_net_u(x_test, y_test)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1, 1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1, 1)
        # Error
        error_u = np.linalg.norm(u_test - u_pred, 2) / np.linalg.norm(u_test, 2)
        error_v = np.linalg.norm(v_test - v_pred, 2) / np.linalg.norm(v_test, 2)
        return error_u, error_v

    def evaluate(self, x, y, u, v, opt_name, loop=True):
        """ testing all points in the domain """
        x_test = x.reshape(-1, 1)
        y_test = y.reshape(-1, 1)
        u_test = u.reshape(-1, 1)
        v_test = v.reshape(-1, 1)
        # Prediction
        x_test = torch.tensor(x_test, requires_grad=True).float().to(device)
        y_test = torch.tensor(y_test, requires_grad=True).float().to(device)
        u_pred, v_pred, p_pred = self.neural_net_u(x_test, y_test)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1, 1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1, 1)
        p_pred = p_pred.detach().cpu().numpy().reshape(-1, 1)
        # Error
        error_u = np.linalg.norm(u_test - u_pred, 2) / np.linalg.norm(u_test, 2)
        error_v = np.linalg.norm(v_test - v_pred, 2) / np.linalg.norm(v_test, 2)
        print('------------------------')
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))

        if loop:
            div_pred = self.divergence(x_test, y_test)
            u_pred = u_pred.reshape(257, 257)
            v_pred = v_pred.reshape(257, 257)
            p_pred = p_pred.reshape(257, 257)
            div_pred = div_pred.reshape(257, 257)

            Re_folder = 'Re' + str(self.Re)

            save_results_to = 'results/' + Re_folder + '/'

            if not os.path.exists(save_results_to):
                os.makedirs(save_results_to)

            scipy.io.savemat(save_results_to + 'cavity_result_' + opt_name + '.mat',
                             {
                                 'U_pred': u_pred,
                                 'V_pred': v_pred,
                                 'P_pred': p_pred,
                                 'div_pred': div_pred,
                              })

    def divergence(self, x_star, y_star):
        (self.eq1_pred, self.eq2_pred,
         self.eq3_pred) = self.neural_net_equations(x_star, y_star)
        div = self.eq3_pred
        return div
