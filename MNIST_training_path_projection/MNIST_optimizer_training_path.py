import torch
from torchvision import transforms
from torch.optim import *
from models import *
import numpy as np

from torchvision import datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import os
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

from PIDAO_SI import PIDAccOptimizer_SI
from PIDAO_Sym_Tz import PIDAccOptimizer_Sym_Tz
from PIDAO_SI_AAdRMS import PIDAccOptimizer_SI_AAdRMS

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
batch_size = 128
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
num_epochs = 200


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

    def init_weight(self, a, b, pcacoords):
        t = self.state_dict()
        weight_list, _ = pcacoords.init_matrix(a=a, b=b)
        num = 0
        for key, value in self.state_dict().items():
            if 'weight' in key:
                w = torch.tensor(weight_list[num])
                t[key].copy_(w)
                num += 1

    def init_bias(self, a, b, pcacoords):
        t = self.state_dict()
        _, bias_list = pcacoords.init_matrix(a=a, b=b)
        num = 0
        for key, value in self.state_dict().items():
            if 'bias' in key:
                w = torch.tensor(bias_list[num])
                t[key].copy_(w)
                num += 1

    def get_matrix(self):
        # Collecting each weight of this net
        weight_list = []
        for name, parameters in self.state_dict().items():
            if 'weight' in name:
                weight_list.append(parameters.cpu().clone().numpy())

        # Collecting each bias of this net
        bias_list = []
        for name, parameters in self.state_dict().items():
            if 'bias' in name:
                bias_list.append(parameters.cpu().clone().numpy())
        return weight_list, bias_list


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

    Matrix = model.get_matrix()

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
                  % (epoch + 1, num_epochs, batch + 1, len(train_data), train_loss_log.avg,
                     train_acc_log.avg))
    scheduler.step()


    return train_loss_log, train_acc_log, Matrix


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
    equivalent_momentum = 0.9
    momentum = (1 / equivalent_momentum - 1) / lr
    ki = 0.1
    kd = 1
    kp = 1 * lr * (1 + momentum * lr) / lr ** 2
    
    p_ar_lr = alr
    momentum_ar = (1 / equivalent_momentum - 1) / p_ar_lr
    ki_ar = 4
    kd_ar = 1
    kp_ar = 1 * p_ar_lr * (1 + momentum_ar * p_ar_lr) / p_ar_lr ** 2
    # a collection of all optimizers for comparisons
    optimizers = {
        'Adam': Adam(model.parameters(), lr=1*alr, weight_decay=0.0001),
        'AdamW': AdamW(model.parameters(), lr=alr, weight_decay=0.0001),
        'SGDM': SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9),
        'SGDMN': SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9, nesterov=True),
        'PIDAO_Sym_Tz': PIDAccOptimizer_Sym_Tz(model.parameters(), lr=lr,
                                               weight_decay=0.0001, momentum=momentum, kp=kp, ki=ki, kd=kd),
        'PIDAO_SI': PIDAccOptimizer_SI(model.parameters(), lr=lr,
                                       weight_decay=0.0001, momentum=momentum, kp=kp, ki=ki, kd=kd)
        'PIDAO_SI_AAdRMS': PIDAccOptimizer_SI_AAdRMS(model.parameters(), lr=p_ar_lr, weight_decay=0.0001, momentum=momentum_ar, kp=kp_ar, ki=ki_ar, kd=kd_ar)
    }
    return optimizers[method_name]


def main(train_data, test_data, model, loss_fn, optimizer, num_epochs, logger, device, scheduler, training_path):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loss_log, train_acc_log, Matrix \
            = \
            train_loop(train_data=train_data,
                       model=model,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       epoch=epoch,
                       device=device,
                       scheduler=scheduler,
                       )
        val_loss_log, val_acc_log = test_loop(test_data=test_data,
                                              model=model,
                                              loss_fn=loss_fn,
                                              device=device)
        training_path.append(Matrix)
        logger.append([optimizer.param_groups[0]['lr'],
                       train_loss_log.avg,
                       val_loss_log.avg,
                       train_acc_log.avg,
                       val_acc_log.avg])
    logger.close()
    logger.plot()
    return training_path


if __name__ == '__main__':
    # learning_rate = float(input('Your expected learning rate of optimizers is:'))
    learning_rate = 0.03
    path = 'results/MNIST_training_path' + '/learning_rate={0}'.format(learning_rate)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

    # net's initialization
    NN_set = {'FNN': Net()}
    NN = 'FNN'
    initial_net = NN_set[NN]
    path_nn = path + '/NN={0}'.format(NN)
    if not os.path.exists(path_nn):
        os.makedirs(path_nn)
    # save the net's structure
    torch.save(initial_net, path_nn + '/initial_net.pkl')

    method = ['Adam', 'SGDM', 'PIDAO_Sym_Tz', 'PIDAO_SI', 'PIDAO_SI_AAdRMS']
    for optimizer_name in method:
        logger = Logger(path_nn + '/' + optimizer_name + '.txt')
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        net = torch.load(path_nn + '/initial_net.pkl')
        # GPU or CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        loss = nn.CrossEntropyLoss()
        opt = Optimizers(net, optimizer_name, learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=50, gamma=1)
        print('Here is a training process driven by the {0} optimizer'.format(optimizer_name))
        training_path = []
        training_path \
            = \
            main(train_data=train_loader,
                 test_data=test_loader,
                 model=net,
                 loss_fn=loss,
                 optimizer=opt,
                 num_epochs=num_epochs,
                 logger=logger,
                 device=device,
                 scheduler=scheduler,
                 training_path=training_path
                 )
        np.save(path_nn + '/training_path_' + optimizer_name + '.npy', np.array(training_path, dtype=object))