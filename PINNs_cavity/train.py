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
from tools import *
import cavity_data as cavity
import pinn_solver as psolver
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
import os

from PIDAO_SI_AAdRMS import PIDAccOptimizer_SI_AAdRMS
p_ar_lr = 1e-3
equivalent_momentum = 0.9
momentum_ar = (1 / equivalent_momentum - 1) / p_ar_lr
kp = 2
ki_ar = 4
kd_ar = .01
kp_ar = kp * p_ar_lr * (1 + momentum_ar * p_ar_lr) / p_ar_lr ** 2


def train(opt_name, net_params=None):
    Re = 100   # Reynolds number
    # N_HLayer = 3
    N_HLayer = 6
    N_neu = 80
    layers = [2] + N_HLayer*[N_neu] + [3]
    lam_bcs = 1
    lam_equ = 1


    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re,
        layers=N_HLayer,
        num_node = N_neu, 
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        checkpoint_path='./checkpoint/',
        net_params=net_params)

    path = './data/'
    dataloader = cavity.DataLoader(path=path, N_f=5000)

    # Set boundary data, | u, v, x, y
    boundary_data = dataloader.loading_boundary_data()
    PINN.set_boundary_data(X=boundary_data)

    # Set training data, | x, y
    training_data = dataloader.loading_training_data()
    PINN.set_eq_training_data(X=training_data)

    filename = './data/cavity_Re'+str(Re)+'.mat'
    x_star, y_star, u_star, v_star = dataloader.loading_evaluate_data(filename)
    
    test_set = [x_star, y_star, u_star, v_star]
    
    Re_folder = 'Re' + str(PINN.Re)
    path = 'results/' + Re_folder + '/'
    if not os.path.exists(path):
      os.makedirs(path)
    
    if opt_name == 'Adam':
      opt = torch.optim.Adam(params=PINN.net.parameters(), lr=kp*p_ar_lr, weight_decay=0)
    elif opt_name == 'AdamW':
      opt = torch.optim.AdamW(params=PINN.net.parameters(), lr=kp*p_ar_lr, weight_decay=0)
    elif opt_name == 'RMSprop':
      opt = torch.optim.RMSprop(params=PINN.net.parameters(),lr=kp*p_ar_lr,alpha=0.99,eps=1e-08,weight_decay=0,momentum=0,centered=False)
    elif opt_name == 'PIDAO':
      opt = PIDAccOptimizer_SI_AAdRMS(params=PINN.net.parameters(), lr=p_ar_lr, weight_decay=0,momentum=momentum_ar, kp=kp_ar, ki=ki_ar, kd=kd_ar)
    elif opt_name == 'SGDM':
      opt = torch.optim.SGD(params=PINN.net.parameters(), lr=kp*p_ar_lr, weight_decay=0, momentum=0.9)
    
    logger = Logger(path + opt_name + '.txt')
    logger.set_names(['Learning Rate', 'Loss', 'Eq Loss', 'BC Loss', 'Error u', 'Error v'])
    
    print("Here is the training process by the " + opt_name)
    PINN.train(num_epoch=100000, test_set=test_set, logger=logger, optimizer=opt)
    PINN.evaluate(x_star, y_star, u_star, v_star, opt_name=opt_name)
    
    
if __name__ == "__main__":
    # train(opt_name='AdamW')
    # train(opt_name='PIDAO')
    # train(opt_name='Adam')
    # train(opt_name='RMSprop')
    train(opt_name='SGDM')
