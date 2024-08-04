import torch
from torchvision import transforms
from torch.optim import *
from models import *

from torchvision import datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import os
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
import time

from PIDAO_SI import PIDAccOptimizer_SI
from PIDAO_Sym_Tz import PIDAccOptimizer_Sym_Tz
from PIDAO_SI_AAdRMS import PIDAccOptimizer_SI_AAdRMS
from PIDopt import PIDOptimizer
from AdaHB import Adaptive_HB


import argparse
# args configurationz
parser = argparse.ArgumentParser(description='FashionMNIST Example Config')
parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--ki', type=float, default=0.3, help='ki')
parser.add_argument('--kd', type=float, default=1, help='kd')
parser.add_argument('--gpu', type=int, default=0, help='the number of gpu for training')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--num-epochs', type=int, default=2, help='number of epochs to train')
parser.add_argument('--model', type=str, default='FNN', help='model type')
args = parser.parse_args()

# MNIST Dataset
training_data = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor()
                               )

test_data = datasets.MNIST(root='./data',
                           train=False,
                           download=True,
                           transform=transforms.ToTensor()
                           )

# Data Loader (Input Pipeline)
batch_size = args.batch_size
train_loader = DataLoader(dataset=training_data,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2
                          )

test_loader = DataLoader(dataset=test_data,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=2
                         )

# Hyper Parameters
input_size = 28 * 28
hidden_size = 1000
num_classes = 10
num_epochs = args.num_epochs


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=1000, output_size=10):
        super(Net, self).__init__()
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
        logits = self.linear_relu_stack(x)
        return logits


class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=1)
        self.maxpooling1 = nn.MaxPool2d(2)
        self.maxpooling2 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(360, 150)
        self.fc2 = nn.Linear(150, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)  # 256*20*26*26
        x = self.maxpooling1(x)  # 256*20*13*13
        x = self.relu(x)
        x = self.conv2(x)  # 256*40*9*9
        x = self.maxpooling2(x)  # 256*40*3*3
        x = self.relu(x)
        x = x.view(in_size, -1)  # 256*360
        x = self.fc1(x)  # 256*150
        x = self.dropout(x)
        x = self.fc2(x)  # 256*10
        return x


# Training process
def train_loop(train_data, model, loss_fn, optimizer, epoch, device, scheduler):
    r"""
    :param train_data: train_loader
    :param model: trained model
    :param loss_fn: loss function of the model
    :param optimizer: optimizer
    :param epoch: current epoch
    :param device: computation by GPU or CPU
    :return: the training loss and accuracy under this epoch
    """
    model.train()
    train_loss_log = AverageMeter()
    train_acc_log = AverageMeter()
    for batch, (images, labels) in enumerate(train_data):
        # Convert torch tensor to Variable
        images = images.to(device)
        labels = labels.to(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = model(images)
        train_loss = loss_fn(outputs, labels)
        train_loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        train_loss_log.update(train_loss.item(), images.size(0))
        train_acc_log.update(prec1.item(), images.size(0))

        if (batch + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.8f'
                  % (epoch + 1, args.num_epochs, batch + 1, len(train_data), train_loss_log.avg,
                     train_acc_log.avg))
    scheduler.step()

    return train_loss_log, train_acc_log


def test_loop(test_data, model, loss_fn, device):
    r"""
    :param test_data: test_loader
    :param model: trained model
    :param loss_fn: loss function of the model
    :param device: computation by GPU or CPU
    :return: the test loss and accuracy under a specific epoch
    """
    val_loss_log = AverageMeter()
    val_acc_log = AverageMeter()
    model.eval()
    for images, labels in test_data:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_loss = loss_fn(outputs, labels)
        val_loss_log.update(test_loss.item(), images.size(0))
        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        val_acc_log.update(prec1.item(), images.size(0))

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
    alr = 1e-3  # for adaptive optimizers
    equivalent_momentum = args.momentum
    momentum = (1 / equivalent_momentum - 1) / lr
    # ki = 0.1
    ki = 0.1
    kd = 1
    kp = 1 * lr * (1 + momentum * lr) / lr ** 2
    
    p_ar_lr = 1e-3
    momentum_ar = (1 / equivalent_momentum - 1) / p_ar_lr
    ki_ar = 4
    kd_ar = 1
    kp_ar = 1 * p_ar_lr * (1 + momentum_ar * p_ar_lr) / p_ar_lr ** 2

    # a collection of all optimizers for comparisons
    optimizers = {
        'Adam': Adam(model.parameters(), lr=1*alr, weight_decay=0.0001),
        'AdamW': AdamW(model.parameters(), lr=alr, weight_decay=0.0001),
        'Adadelta': Adadelta(model.parameters(), lr=alr, weight_decay=0.0001),
        'RMSprop': RMSprop(model.parameters(), lr=alr, weight_decay=0.0001),
        'SGDM': SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9),
        'PIDAO_Sym_Tz': PIDAccOptimizer_Sym_Tz(model.parameters(), lr=lr,
                                               weight_decay=0.0001, momentum=momentum, kp=kp, ki=ki, kd=kd),
        'PIDAO_SI': PIDAccOptimizer_SI(model.parameters(), lr=lr,
                                       weight_decay=0.0001, momentum=momentum, kp=kp, ki=ki, kd=kd),
        'PIDAO_SI_AAdRMS': PIDAccOptimizer_SI_AAdRMS(model.parameters(), lr=p_ar_lr, weight_decay=0.0001, momentum=momentum_ar, kp=kp_ar, ki=ki_ar, kd=kd_ar),
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
        logger.append([optimizer.param_groups[0]['lr'], 
                       train_loss_log.avg, 
                       val_loss_log.avg, 
                       train_acc_log.avg, 
                       val_acc_log.avg, 
                       execution_time])
    logger.close()
    logger.plot()


if __name__ == '__main__':
    # learning_rate = float(input('Your expected learning rate of optimizers is:'))
    learning_rate = args.lr
    path = 'results/mnist' + '/learning_rate={0}'.format(learning_rate)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

    # net's initialization
    NN_set = {'FNN': Net(), 'CNN': CNet()}
    # NN = input('There are fully connected networks (input FNN) and convolutional networks (input CNN) to choose from'
    #            '\n Your expected neural network model for the MNIST is: ')
    NN = args.model
    initial_net = NN_set[NN]
    path_nn = path + '/NN={0}'.format(NN)
    if not os.path.exists(path_nn):
        os.makedirs(path_nn)
    # save the net's structure
    torch.save(initial_net, path_nn + '/initial_net.pkl')

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
        logger = Logger(path_nn + '/' + optimizer_name + '.txt')
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Time'])
        net = torch.load(path_nn + '/initial_net.pkl')
        # GPU or CPU
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        net.to(device)
        loss = nn.CrossEntropyLoss()
        opt = Optimizers(net, optimizer_name, learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=50, gamma=1)
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
