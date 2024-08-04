import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *
import scipy
from logger import Logger

from PIDAO_SI_AAdRMS import PIDAccOptimizer_SI_AAdRMS
from AdaHB import Adaptive_HB


# torch.manual_seed(0)
# np.random.seed(0)


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


def Optimizers(model, opt_name, learning_rate):
    p_ar_lr = learning_rate
    equivalent_momentum = 0.7
    momentum_ar = (1 / equivalent_momentum - 1) / p_ar_lr
    kp = 4
    ki_ar = 4
    kd_ar = 1
    kp_ar = kp * p_ar_lr * (1 + momentum_ar * p_ar_lr) / p_ar_lr ** 2
    Optimizers = {
        'Adam': torch.optim.Adam(model.parameters(), lr=kp*learning_rate, weight_decay=1e-4),
        'AdamW': torch.optim.AdamW(model.parameters(), lr=kp*p_ar_lr, weight_decay=1e-4),
        'RMSprop': torch.optim.RMSprop(model.parameters(),lr=kp*p_ar_lr,alpha=0.99,eps=1e-08,weight_decay=1e-4,momentum=0,centered=False),
        'PIDAO': PIDAccOptimizer_SI_AAdRMS(model.parameters(), lr=p_ar_lr, weight_decay=1e-4,momentum=momentum_ar, kp=kp_ar, ki=ki_ar, kd=kd_ar
        ),
        'AdaHB': Adaptive_HB(model.parameters(), lr=kp*p_ar_lr, weight_decay=1e-4, momentum_init=0.9),
    }
    return Optimizers[opt_name]


def FNO_main(train_data_res, save_index, logger, optimizer, model):
    """
    Parameters
    ----------
    train_data_res : resolution of the training data
    save_index : index of the saving folder
    """

    ################################################################
    # configs
    ################################################################
    TRAIN_PATH = 'datasets_FNO/piececonst_r421_N1024_smooth1.mat'
    TEST_PATH = 'datasets_FNO/piececonst_r421_N1024_smooth1.mat'

    ntrain = 1000  # first 1000 of smooth1.mat
    ntest = 100  # first 100 of smooth1.mat

    batch_size = 20

    epochs = 300
    step_size = 100
    gamma = .6

    s = train_data_res
    r = (421 - 1) // (s - 1)

    ################################################################
    # load data and data normalization
    ################################################################
    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]  # * 0.1 - 0.75
    y_train = reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s]  # * 100

    reader.load_file(TEST_PATH)
    x_test = reader.read_field('coeff')[-ntest:, ::r, ::r][:, :s, :s]  # * 0.1 - 0.75
    y_test = reader.read_field('sol')[-ntest:, ::r, ::r][:, :s, :s]  # * 100

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    grids = []
    grid_all = np.linspace(0, 1, 421).reshape(421, 1).astype(np.float64)
    grids.append(grid_all[::r, :])
    grids.append(grid_all[::r, :])
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1, s, s, 2)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), grid.repeat(ntrain, 1, 1, 1)], dim=3)
    x_test = torch.cat([x_test.reshape(ntest, s, s, 1), grid.repeat(ntest, 1, 1, 1)], dim=3)

    # x_train = x_train.reshape(ntrain,s,s,1)
    # x_test = x_test.reshape(ntest,s,s,1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                              shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    model = model
    print(count_params(model))

    optimizer = optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    start_time = default_timer()
    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_mse = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x).reshape(batch_size, s, s)

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            # mse.backward()

            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            loss.backward()

            optimizer.step()
            train_mse += mse.item()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x).reshape(batch_size, s, s)
                out = y_normalizer.decode(out)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        # print(ep, t2-t1, train_l2, test_l2)
        print("Epoch: %d, time: %.3f, Train Loss: %.3e, Train l2: %.4f, Test l2: %.4f"
              % (ep, t2 - t1, train_mse, train_l2, test_l2))
        logger.append([ep, t2 - t1, train_mse, train_l2, test_l2])

    elapsed = default_timer() - start_time
    print("\n=============================")
    print("Training done...")
    print('Training time: %.3f' % (elapsed))
    print("=============================\n")


if __name__ == "__main__":

    training_data_resolution = 29
    save_index = 0

    path = 'results/'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

    modes = 12
    width = 32
    model = FNO2d(modes, modes, width).cuda()
    torch.save(model, path + 'initial_net.pkl')

    learning_rate = 1e-3

    Opt_set = [
        # 'PIDAO', 'Adam', 
        'AdamW',
        # 'RMSprop', 'AdaHB'
               ]
    for opt_name in Opt_set:
        model = torch.load(path + 'initial_net.pkl')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger = Logger(path + '/' + opt_name + '.txt')
        logger.set_names(['Epoch', 'time', 'Train Loss', 'Train L_2', 'Test L_2'])
        opt = Optimizers(model, opt_name, learning_rate)
        print('Here is a training process driven by the {0} optimizer'.format(opt_name))
        FNO_main(training_data_resolution, save_index, logger, opt, model)










