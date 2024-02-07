from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig


# ### Code for processing weights ### #
def vectorize_weights_(weights):
    r'''
    :return: Converting a two-dimension vector into a one-dimension vector
    '''
    vec = [w.flatten() for w in weights]
    vec = np.hstack(vec)
    return vec


def vectorize_weight_list_(weight_list):
    r'''
    :param weight_list: a list including all weight matrices of each layer,
           where each element of this list is a two-dimension vector
    :return: Converting this list into a two-dimension vector with only row
    e.g.: weight_list = [np.array([[1,2,3],[3,4,5]]), np.array([[1,2],[2,3],[3,4],[4,5]])]
    # >>> array([[1, 2, 3, 3, 4, 5, 1, 2, 2, 3, 3, 4, 4, 5]]
    '''
    vec_list = []
    weight_size_list = []
    for weights in weight_list:
        weight_size_list.append(weights.shape)
        vec_list.append(vectorize_weights_(weights))
    weight_matrix = vec_list[0]
    for k in range(1, len(vec_list)):
        weight_matrix = np.append(weight_matrix, vec_list[k])
    weight_matrix = np.expand_dims(weight_matrix, axis=0)
    return weight_matrix, weight_size_list


def reshape_weight_matrix_list_(weight_matrix, weight_size_list):
    r'''
    :param weight_matrix: a two-dimension vector with only row
    :return: Converting this weight_matrix into a weight_list following the rule from the size_list
    '''
    weight_list = []
    index = 0
    for size in weight_size_list:
        length = size[0] * size[1]
        weight = weight_matrix[..., index: (index + length)]
        index += length
        weight_list.append(weight.reshape(size))
    return weight_list


# ### Code for processing biases ### #
def vectorize_bias_list_(bias_list):
    r'''
    :param bias_list: a list including all bias matrices of each layer,
            where each element of this list is a one-dimension vector
    :return: Converting this list into a two-dimension vector with only row.
    # e.g.: bias_list = [np.array([1,2,3]), np.array([1,2])]
    # >>> array([[1, 2, 3, 1, 2]]
    '''
    vec_list = []
    bias_size_list = []
    for bias in bias_list:
        bias_size_list.append(bias.shape)
        vec_list.append(bias)
    bias_matrix = vec_list[0]
    for k in range(1, len(vec_list)):
        bias_matrix = np.append(bias_matrix, vec_list[k])
    bias_matrix = np.expand_dims(bias_matrix, axis=0)
    return bias_matrix, bias_size_list


def reshape_bias_matrix_list_(bias_matrix, bias_size_list):
    r'''
    :param bias_matrix: The bias_matrix is a two-dimension vector with only row
    :return: Converting this bias_matrix into a bias_list following the rule from the size_list
    '''
    bias_list = []
    index = 0
    for size in bias_size_list:
        length = size[0]
        bias = bias_matrix[..., index: (index + length)]
        index += length
        bias_list.append(bias.reshape(size))
    return bias_list


# ### Code for processing weights and biases list ### #
def vectorize_weight_and_bias_list_(training_matrix):
    r'''
    :param training_matrix: The input, training_matrix, includes two lists, i.e., weight_list and bias_list
    :return: Converting two lists into a two-dimension vector with only one row
    '''
    weight_list, bias_list = training_matrix
    weight_matrix, weight_size_list = vectorize_weight_list_(weight_list)
    bias_matrix, bias_size_list = vectorize_bias_list_(bias_list)
    para_matrix = np.concatenate((weight_matrix, bias_matrix), axis=1)
    return para_matrix


def reshape_vec(vec, weight_size_list, bias_size_list):
    weight_num = 0
    for k in weight_size_list:
        weight_num += k[0] * k[1]
    weight_matrix = vec[..., 0:weight_num]
    bias_matrix = vec[..., weight_num:None]
    vec_weight = reshape_weight_matrix_list_(weight_matrix, weight_size_list)
    vec_bias = reshape_bias_matrix_list_(bias_matrix, bias_size_list)
    return vec_weight, vec_bias


def normalize_weights(weights, origin):
    return [
        w * np.linalg.norm(wc) / np.linalg.norm(w)
        for w, wc in zip(weights, origin)
    ]


# ### PCA coordinates ### #
class PCACoordinates(object):
    def __init__(self, training_path, weight_size_list, bias_size_list, pca_project=True):
        r"""
        training_path: the iterative point by training neural networks from every epoch
        weight_size_list: the k_th item in this list records the weight size of the k_th layer of the NN
        bias_size_list: the k_th item in this list records the bias size of the k_th layer of the NN
        pca_project: True means to obtain two direction by the pca projectio,
                    and vice versa means that two directions are obtained by random initialization
        """
        self.origin_ = vectorize_weight_and_bias_list_(training_path[-1])
        self.training_data_ = np.empty([len(training_path) - 1, self.origin_.shape[1]])
        self.weight_size_list = weight_size_list
        self.bias_size_list_ = bias_size_list

        for k, training_matrix in enumerate(training_path[0:-1]):
            para_matrix = vectorize_weight_and_bias_list_(training_matrix)
            para_matrix -= self.origin_
            self.training_data_[k] = para_matrix

        if pca_project:
            pca = PCA(n_components=2)
            self.coordinates_ = pca.fit_transform(self.training_data_)
            self.variance_ratio_ = pca.explained_variance_ratio_

            # get the PCA direction (two-dimension vector with only one row)
            self.v0_ = np.expand_dims(pca.components_[0], axis=0)
            self.v1_ = np.expand_dims(pca.components_[1], axis=0)

            # list composing by the weight/bias matrix from each layer
            self.origin_weight, self.origin_bias = reshape_vec(self.origin_, self.weight_size_list,
                                                               self.bias_size_list_)
            self.v0_weight, self.v0_bias = reshape_vec(self.v0_, self.weight_size_list, self.bias_size_list_)
            self.v1_weight, self.v1_bias = reshape_vec(self.v1_, self.weight_size_list, self.bias_size_list_)
            
            # filter normalization
            for k, origin in enumerate(self.origin_weight):
                self.v0_weight[k] = np.array(normalize_weights(self.v0_weight[k], origin))
                self.v1_weight[k] = np.array(normalize_weights(self.v1_weight[k], origin))
            for k, bias in enumerate(self.origin_bias):
                self.v0_bias[k] = np.array(normalize_weights(self.v0_bias[k], bias))
                self.v1_bias[k] = np.array(normalize_weights(self.v1_bias[k], bias))
        else:
            # get the random direction (two-dimension vector with only one row)
            self.v0_ = np.random.normal(size=self.origin_.shape)
            self.v1_ = np.random.normal(size=self.origin_.shape)

            # list composing by the weight/bias matrix from each layer
            self.origin_weight, self.origin_bias = reshape_vec(self.origin_, self.weight_size_list,
                                                               self.bias_size_list_)
            self.v0_weight, self.v0_bias = reshape_vec(self.v0_, self.weight_size_list, self.bias_size_list_)
            self.v1_weight, self.v1_bias = reshape_vec(self.v1_, self.weight_size_list, self.bias_size_list_)

            # filter normalization
            for k, origin in enumerate(self.origin_weight):
                self.v0_weight[k] = np.array(normalize_weights(self.v0_weight[k], origin))
                self.v1_weight[k] = np.array(normalize_weights(self.v1_weight[k], origin))
            for k, bias in enumerate(self.origin_bias):
                self.v0_bias[k] = np.array(normalize_weights(self.v0_bias[k], bias))
                self.v1_bias[k] = np.array(normalize_weights(self.v1_bias[k], bias))

    def get_projected_coordinates(self):
        v0 = vectorize_weight_and_bias_list_((self.v0_weight, self.v0_bias))
        v1 = vectorize_weight_and_bias_list_((self.v1_weight, self.v1_bias))
        A = np.append(v0, v1, axis=0)
        b = self.training_data_

        # Solving the least-square problem formulated by 'b=XA', where X is the solution formulated by {(AA^T)^-1Ab^T}^T
        X = (np.linalg.inv(np.dot(A, A.T)) @ A @ b.T).T
        x = X[..., 0]
        y = X[..., 1]
        return x, y

    def get_variance_ratio(self):
        return self.variance_ratio_

    def init_matrix_2D(self, a, b):
        weight_list, bias_list = [], []
        for k in range(len(self.origin_weight)):
            weight_list.append(self.origin_weight[k] + a * self.v0_weight[k] + b * self.v1_weight[k])
        for k in range(len(self.origin_bias)):
            bias_list.append(self.origin_bias[k] + a * self.v0_bias[k] + b * self.v1_bias[k])
        return weight_list, bias_list

    def init_matrix_1D(self, a):
        weight_list, bias_list = [], []
        for k in range(len(self.origin_weight)):
            weight_list.append(self.origin_weight[k] + a * self.v0_weight[k])
        for k in range(len(self.origin_bias)):
            bias_list.append(self.origin_bias[k] + a * self.v0_bias[k])
        return weight_list, bias_list


# ### Loss landscape ### #
class LossSurface(object):
    def __init__(self, model, train_loader, test_loader):
        self.model_ = model
        inputs_train, outputs_train = next(iter(train_loader))
        self.inputs_train = inputs_train
        self.outputs_train = outputs_train
        inputs_test, outputs_test = next(iter(test_loader))
        self.inputs_test = inputs_test
        self.outputs_test = outputs_test
        self.train_loader = train_loader
        self.test_loader = test_loader

    # ## Drawing the two-dimensional projection picture ## #
    # Calculating the loss value f(a, b) = L(\theta^{\star} + a*\theta_1 + b*\theta_2)
    def compile_2D(self, range, points, pcacoords):
        r"""
        range: the projection range. For example, range=2 means that the projection range is [-2, 2]x[-2, 2]
        points: the number of samples on each projection component.
        pcacoords: the PCA coordinate processor
        """
        a_grid = np.linspace(-1.0, 1.0, num=points) ** 1 * range
        b_grid = np.linspace(-1.0, 1.0, num=points) ** 1 * range
        loss_grid = np.empty([len(a_grid), len(b_grid)])
        for i, a in enumerate(a_grid):
            for j, b in enumerate(b_grid):
                # reset the weights of this net
                self.model_.init_weight_2D(a=a, b=b, pcacoords=pcacoords)
                self.model_.init_bias_2D(a=a, b=b, pcacoords=pcacoords)
                # the output of this net with input equaling to x
                xx = self.inputs_train.to(device)
                yy = self.outputs_train.to(device)
                yy_pre = self.model_(xx)
                loss = nn.CrossEntropyLoss()
                loss = loss(yy_pre, yy)

                loss_grid[j, i] = loss.item()
        self.model_.init_weight_2D(a=0, b=0, pcacoords=pcacoords)
        self.model_.init_bias_2D(a=0, b=0, pcacoords=pcacoords)
        self.a_grid_ = a_grid
        self.b_grid_ = b_grid
        self.loss_grid_ = loss_grid

    # Ploting the two-dimensional projection in a contour picture.
    def plot_2D(self, range=1.0, points=24, level=20, ax=None, **kwargs):
        xs = self.a_grid_
        ys = self.b_grid_
        zs = self.loss_grid_
        if ax is None:
            _, ax = plt.subplots(**kwargs)
            ax.set_title("The Loss Surface")
            ax.set_aspect("equal")
        else:
            ax.set_title("The Loss Surface")
        # Set Levels
        min_loss = zs.min()
        max_loss = zs.max()
        levels = np.exp(
            np.linspace(
                float(np.log(min_loss)), float(np.log(max_loss)), num=level
            )
        )
        # Create Contour Plot
        CS = ax.contour(
            xs,
            ys,
            zs,
            levels=levels,
            cmap="magma",
            linewidths=0.75,
            norm=matplotlib.colors.LogNorm(vmin=min_loss, vmax=max_loss * 2.0),
        )
        ax.clabel(CS, inline=True, fontsize=8, fmt="%1.2f")
        return ax

    # ## Drawing the one-dimensional projection picture ## #
    # Calculating the loss value f(a) = L(\theta^{\star} + a*\theta_1)
    def compile_1D(self, points, pcacoords):
        r"""
        range: the projection range. For example, range=2 means that the projection range is [-2, 2]
        points: the number of samples on the projection component.
        pcacoords: the PCA coordinate processor
        """
        a_grid = np.linspace(-1.0, 1.0, num=points) ** 1
        train_loss_grid = np.empty(len(a_grid))
        test_loss_grid = np.empty(len(a_grid))
        train_acc_grid = np.empty(len(a_grid))
        test_acc_grid = np.empty(len(a_grid))
        for i, a in enumerate(a_grid):
            # reset the weights of this net
            self.model_.init_weight_1D(a=a, pcacoords=pcacoords)
            self.model_.init_bias_1D(a=a, pcacoords=pcacoords)

            
            # the output of this net with input equaling to x
            train_loss_log = AverageMeter()
            train_acc_log = AverageMeter()
            for inputs_train, outputs_train in self.train_loader:
                xx_train = inputs_train.to(device)
                yy_train = outputs_train.to(device)
                yy_pre = self.model_(xx_train)
                loss_fn = nn.CrossEntropyLoss()
                train_loss = loss_fn(yy_pre, yy_train)
                train_loss_log.update(train_loss.item(), xx_train.size(0))
                prec1, prec5 = accuracy(yy_pre.data, yy_train.data, topk=(1, 5))
                train_acc_log.update(prec1.item(), xx_train.size(0))
            train_acc_grid[i] = train_acc_log.avg
            train_loss_grid[i] = train_loss_log.avg

            val_loss_log = AverageMeter()
            val_acc_log = AverageMeter()
            for inputs_test, outputs_test in self.test_loader:
                xx_test = inputs_test.to(device)
                yy_test = outputs_test.to(device)
                outputs = self.model_(xx_test)
                test_loss = loss_fn(outputs, yy_test)
                val_loss_log.update(test_loss.item(), xx_test.size(0))
                prec1, prec5 = accuracy(outputs.data, yy_test.data, topk=(1, 5))
                val_acc_log.update(prec1.item(), xx_test.size(0))
            test_loss_grid[i] = val_loss_log.avg
            test_acc_grid[i] = val_acc_log.avg


        self.model_.init_weight_1D(a=0, pcacoords=pcacoords)
        self.model_.init_bias_1D(a=0, pcacoords=pcacoords)
        self.aa_grid_ = a_grid
        self.train_lloss_grid_ = train_loss_grid
        self.train_aacc_grid_ = train_acc_grid
        self.test_lloss_grid_ = test_loss_grid
        self.test_aacc_grid_ = test_acc_grid

    # Ploting the one-dimensional projection in a picture.
    def plot_1D(self, ax=None, **kwargs):
        xs = self.aa_grid_
        ys = self.train_lloss_grid_
        if ax is None:
            _, ax = plt.subplots(**kwargs)
            ax.set_title("The Loss Surface")
        else:
            ax.set_title("The Loss Surface")
        ax.plot(xs, ys)
        ax.plot(xs, self.test_lloss_grid_)
        ax_adjoint = ax.twinx()
        ax_adjoint.plot(xs, self.train_aacc_grid_)
        ax_adjoint.plot(xs, self.test_aacc_grid_)
        return ax


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
        y = self.linear_relu_stack(x)
        return y

    # initialization of two-dimension ploting
    def init_weight_2D(self, a, b, pcacoords):
        t = self.state_dict()
        weight_list, _ = pcacoords.init_matrix_2D(a=a, b=b)
        num = 0
        for key, value in self.state_dict().items():
            if 'weight' in key:
                w = torch.tensor(weight_list[num])
                t[key].copy_(w)
                num += 1

    def init_bias_2D(self, a, b, pcacoords):
        t = self.state_dict()
        _, bias_list = pcacoords.init_matrix_2D(a=a, b=b)
        num = 0
        for key, value in self.state_dict().items():
            if 'bias' in key:
                w = torch.tensor(bias_list[num])
                t[key].copy_(w)
                num += 1

    # initialization of two-dimension ploting
    def init_weight_1D(self, a, pcacoords):
        t = self.state_dict()
        weight_list, _ = pcacoords.init_matrix_1D(a=a)
        num = 0
        for key, value in self.state_dict().items():
            if 'weight' in key:
                w = torch.tensor(weight_list[num])
                t[key].copy_(w)
                num += 1

    def init_bias_1D(self, a, pcacoords):
        t = self.state_dict()
        _, bias_list = pcacoords.init_matrix_1D(a=a)
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


learning_rate = 0.03
path = 'results/MNIST_training_path' + '/learning_rate={0}'.format(learning_rate)

NN = 'FNN'
path_nn = path + '/NN={0}'.format(NN)

net = Net()
# GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
loss = nn.CrossEntropyLoss()

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
method = ['Adam', 'SGDM', 'PIDAO_Sym_Tz', 'PIDAO_SI', 'PIDAO_SI_AAdRMS']

for optimizer_name in method:
    path_npy = path_nn + '/training_path_' + optimizer_name + '.npy'
    training_path = np.load(path_npy, allow_pickle=True).tolist()

    weight_list, bias_list = training_path[0]
    _, weight_size_list = vectorize_weight_list_(weight_list)
    _, bias_size_list = vectorize_bias_list_(bias_list)

    pcacoords = PCACoordinates(training_path, weight_size_list, bias_size_list, pca_project=True)
    loss_surface = LossSurface(net, train_loader, test_loader)
    loss_surface.compile_2D(range=2, points=100, pcacoords=pcacoords)
    loss_surface.compile_1D(points=100, pcacoords=pcacoords)

    plot_data = {}
    plot_data['a_grid_'] = loss_surface.a_grid_
    plot_data['b_grid_'] = loss_surface.b_grid_
    plot_data['loss_grid_'] = loss_surface.loss_grid_
    plot_data['aa_grid_'] = loss_surface.aa_grid_
    plot_data['train_loss_grid'] = loss_surface.train_lloss_grid_
    plot_data['train_acc_grid'] = loss_surface.train_aacc_grid_
    plot_data['test_loss_grid'] = loss_surface.test_lloss_grid_
    plot_data['test_acc_grid'] = loss_surface.test_aacc_grid_
    xx, yy = pcacoords.get_projected_coordinates()
    plot_data['xx'] = xx
    plot_data['yy'] = yy

    path_dict = path_nn + '/plot_data_PCA_' + optimizer_name + '.npy'
    np.save(path_dict, plot_data)

    # fig = plt.figure(figsize=(10, 5), dpi=100)
    # axes_1 = fig.add_subplot(1, 2, 1)
    # loss_surface.plot_2D(ax=axes_1)
    # axes_1.plot(xx, yy)
    # axes_2 = fig.add_subplot(1, 2, 2)
    # loss_surface.plot_1D(ax=axes_2)
    # plt.show()
    # plt.savefig(path_nn + '/plot_data_PCA_' + optimizer_name + '.pdf', bbox_inches='tight', dpi=600)



