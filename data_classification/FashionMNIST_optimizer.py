import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim import *
from models import *

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from utils import Logger, AverageMeter, accuracy
import time

from PIDAO_SI import PIDAccOptimizer_SI
from PIDAO_Sym_Tz import PIDAccOptimizer_Sym_Tz
from PIDAO_SI_AAdRMS import PIDAccOptimizer_SI_AAdRMS
from PIDopt import PIDOptimizer
from AdaHB import Adaptive_HB

import argparse
# args configurationz
parser = argparse.ArgumentParser(description='FashionMNIST Example Config')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--ki', type=float, default=5, help='ki')
parser.add_argument('--kd', type=float, default=0.9, help='kd')
parser.add_argument('--gpu', type=int, default=0, help='the number of gpu for training')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--model', type=str, default='CNN', help='model type')
args = parser.parse_args()

# FashionMNIST Dataset
training_data = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=ToTensor()
)

# Data Loader (Input Pipeline)
train_loader = DataLoader(training_data,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=2
                          )
test_loader = DataLoader(test_data,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=2
                         )


# Hyper Parameters
input_size = 28*28
hidden_size = 1000
num_classes = 10


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=1000, output_size=10):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Dropout(0.3),
#             nn.BatchNorm2d(64),
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Dropout(0.4),
#         )
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.BatchNorm1d(3 * 3 * 128),
#             nn.Linear(3 * 3 * 128, 256),
#             nn.Dropout(0.25),
#             nn.Linear(256, 10),
#         )

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.fc(out)
#         return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
            )
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x):
        out = self.pool1(self.layer1(x))
        out = self.pool2(self.layer3(self.layer2(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def train_loop(train_data, model, loss_fn, optimizer, epoch, device, scheduler):
    size = len(train_data.dataset)
    model.train()
    train_loss_log = AverageMeter()
    train_acc_log = AverageMeter()
    for batch, (X, y) in enumerate(train_data):

        X = X.to(device)
        y = y.to(device)
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = model(X)
        train_loss = loss_fn(outputs, y)
        train_loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(outputs.data, y.data, topk=(1, 5))
        train_loss_log.update(train_loss.item(), X.size(0))
        train_acc_log.update(prec1.item(), X.size(0))
        if (batch + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.8f'
                  % (epoch + 1, args.num_epochs, batch + 1, len(train_data), train_loss_log.avg,
                     train_acc_log.avg))
    
    scheduler.step()

    return train_loss_log, train_acc_log


def test_loop(test_data, model, loss_fn, device):
    val_loss_log = AverageMeter()
    val_acc_log = AverageMeter()
    model.eval()
    with torch.no_grad():
        for X, y in test_data:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            test_loss = loss_fn(outputs, y)
            val_loss_log.update(test_loss.item(), X.size(0))
            prec1, prec5 = accuracy(outputs.data, y.data, topk=(1, 5))
            val_acc_log.update(prec1.item(), X.size(0))

    print('Accuracy of the network on the 10000 test images: %.8f %%' % val_acc_log.avg)
    print('Loss of the network on the 10000 test images: %.8f' % val_loss_log.avg)

    return val_loss_log, val_acc_log


def Optimizers(model, method_name, learning_rate):
    r"""
    :param model: trained model
    :param method_name: one specific optimizer's name for this model
    :param learning_rate: the learning_rate for optimizers
    :return: one specific optimizer for this model according with the method_name
    """
    # hyper-parameters for PIDAO-series optimizers
    lr = learning_rate
    alr = 2e-3  # for adaptive optimizers
    equivalent_momentum = args.momentum
    momentum = (1 / equivalent_momentum - 1) / lr
    ki = args.ki
    kd = args.kd
    kp = lr * (1 + momentum * lr) / lr ** 2
    
    p_ar_lr = alr
    equivalent_momentum_m = 0.9
    momentum_ar = (1 / equivalent_momentum_m - 1) / p_ar_lr
    # ki_ar = 4
    # kd_ar = 1
    kd_ar = 2*args.kd
    ki_ar = args.ki
    kp_ar = 1 * p_ar_lr * (1 + momentum_ar * p_ar_lr) / p_ar_lr ** 2
    # a collection of all optimizers for comparisons
    optimizers = {
        'Adam': Adam(model.parameters(), lr=alr, weight_decay=0.0001),
        'AdamW': AdamW(model.parameters(), lr=alr, weight_decay=0.0001),
        'RMSprop': RMSprop(model.parameters(), lr=alr, weight_decay=0.0001),
        'SGDM': SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=equivalent_momentum),
        'PIDAO_Sym_Tz': PIDAccOptimizer_Sym_Tz(model.parameters(), lr=lr,
                                               weight_decay=0.0001, momentum=momentum, kp=kp, ki=ki, kd=kd),
        'PIDAO_SI': PIDAccOptimizer_SI(model.parameters(), lr=lr,
                                       weight_decay=0.0001, momentum=momentum, kp=kp, ki=ki, kd=kd),
        'PIDAO_SI_AAdRMS': PIDAccOptimizer_SI_AAdRMS(model.parameters(), lr=p_ar_lr, weight_decay=0.0001,
                                                   momentum=momentum_ar, kp=kp_ar, ki=ki_ar, kd=kd_ar),
        'PIDopt': PIDOptimizer(model.parameters(), lr=lr, weight_decay=0.0001,
                                                   momentum=equivalent_momentum, I=1, D=100),
        'AdaHB': Adaptive_HB(model.parameters(), lr=alr, weight_decay=0.0001, momentum_init=0.9),

    }
    return optimizers[method_name]


def main(train_data, test_data, model, loss_fn, optimizer, num_epochs, logger, device, scheduler):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        start_time = time.time()
        train_loss_log, train_acc_log = train_loop(train_data=train_data,
                                                   model=model,
                                                   loss_fn=loss_fn,
                                                   optimizer=optimizer,
                                                   epoch=epoch,
                                                   device=device,
                                                   scheduler=scheduler)
        end_time = time.time()
        execution_time =  round(end_time - start_time, 3)
        val_loss_log, val_acc_log = test_loop(test_data=test_data,
                                              model=model,
                                              loss_fn=loss_fn,
                                              device=device)
        logger.append([optimizer.param_groups[0]['lr'], train_loss_log.avg, val_loss_log.avg, train_acc_log.avg, val_acc_log.avg, execution_time])
    logger.close()
    logger.plot()


if __name__ == '__main__':
    learning_rate = args.lr
    path = 'results/FashionMNIST' + '/learning_rate={0}'.format(learning_rate)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    # net's initialization
    NN_set = {'FNN': NeuralNetwork(), 'CNN': CNN()}
    NN = args.model
    initial_net = NN_set[NN]
    path_nn = path + '/NN={0}'.format(NN)
    if not os.path.exists(path_nn):
        os.makedirs(path_nn)
    # save the net's structure
    torch.save(initial_net, f'{path_nn}/initial_net.pkl')

    method = [
        'PIDAO_SI', 
        'SGDM', 
        'PIDopt',
        'PIDAO_Sym_Tz',
        'Adam',
        'PIDAO_SI_AAdRMS',
        'AdaHB'
        ]
    for optimizer_name in method:
        logger = Logger(f'{path_nn}/{optimizer_name}.txt')
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Time'])
        net = torch.load(f'{path_nn}/initial_net.pkl')
        # GPU or CPU
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        net.to(device)
        loss = nn.CrossEntropyLoss()
        opt = Optimizers(net, optimizer_name, learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=50, gamma=0.5)
        print('Here is a training process driven by the {0} optimizer'.format(optimizer_name))
        main(train_data=train_loader,
             test_data=test_loader,
             model=net,
             loss_fn=loss,
             optimizer=opt,
             num_epochs=args.num_epochs,
             logger=logger,
             device=device,
             scheduler=scheduler
             )
