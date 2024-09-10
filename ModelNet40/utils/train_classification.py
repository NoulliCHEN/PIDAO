from __future__ import print_function
import argparse
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # this code is for windows platforms
import sys

from AdaHB import Adaptive_HB
from PIDAO_SI_AAdRMS import PIDAccOptimizer_SI_AAdRMS
from PIDopt import PIDOptimizer
from PIDAO_SI import PIDAccOptimizer_SI
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import random
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
from logger import Logger
from timeit import default_timer
from pathlib import Path


def train():

    # adjust the data path and result path
    current_file = Path(__file__)
    modelnet40_source_path = current_file.parents[1]
    opt.dataset = os.path.join(modelnet40_source_path, opt.dataset)
    opt.result_file = os.path.join(modelnet40_source_path, opt.result_file)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # dataset config
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='modelnet10_train')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='modelnet10_test',
        npoints=opt.num_points,
        data_augmentation=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    print(len(dataset), len(test_dataset))
    num_classes = len(dataset.classes)
    print('classes', num_classes)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    # more hyper-parameters pre-config
    equivalent_momentum = opt.momentum
    p_ar_lr = opt.lr
    momentum_ar = (1 / equivalent_momentum - 1) / p_ar_lr
    momentum = (1 / equivalent_momentum - 1) / opt.fixed_lr
    ki = 20
    kd = 1
    kp = 1 * opt.fixed_lr * (1 + momentum * opt.fixed_lr) / opt.fixed_lr ** 2
    ki_ar = 0.1
    kd_ar = 1
    kp_ar = 1 * p_ar_lr * (1 + momentum_ar * p_ar_lr) / p_ar_lr ** 2

    # optimizer (fixed and adaptive)
    optimizer_names = ['Adam', 'RMSprop', 'PIDAO_SI_AAdRMS', 'AdaHB', 'SGDM', 'PIDAO_SI', 'PIDopt']
    for cur_time in range(opt.train_times):
        for opt_name in optimizer_names:
            classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
            optimizers = {
                'Adam': optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=0.0001),
                'RMSprop': optim.RMSprop(classifier.parameters(), lr=opt.lr, weight_decay=0.0001),
                'PIDAO_SI_AAdRMS': PIDAccOptimizer_SI_AAdRMS(classifier.parameters(), lr=opt.lr, weight_decay=0.0001,
                                                             momentum=momentum_ar, kp=kp_ar, ki=ki_ar, kd=kd_ar),
                'AdaHB': Adaptive_HB(classifier.parameters(), lr=opt.lr, weight_decay=0.0001, momentum_init=0.9),
                'SGDM': optim.SGD(classifier.parameters(), lr=opt.fixed_lr, weight_decay=0.0001,
                                  momentum=equivalent_momentum),
                'PIDAO_SI': PIDAccOptimizer_SI(classifier.parameters(), lr=opt.fixed_lr, weight_decay=0.0001,
                                               momentum=momentum, kp=kp, ki=ki, kd=kd),
                'PIDopt': PIDOptimizer(classifier.parameters(), lr=opt.fixed_lr, weight_decay=0.0001,
                                       momentum=equivalent_momentum, I=1, D=100),
            }
            optimizer = optimizers[opt_name]
            classifier.cuda(opt.gpu_id)
            num_batch = len(dataset) / opt.batchSize
            # set result file path
            result_path = os.path.join(opt.result_file, f'modelnet40_{cur_time + 1}')
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            logger = Logger(result_path + '/' + opt_name + '.txt')
            logger.set_names(['Epoch', 'time', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

            # training classification process
            for epoch in range(opt.nepoch):
                start_time = default_timer()
                total_train_correct = 0.
                total_train_testset = 0.
                train_nll_avg_loss = 0.
                train_iter = 0
                for i, data in enumerate(dataloader, 0):
                    train_iter += 1
                    points, target = data
                    target = target[:, 0]
                    points = points.transpose(2, 1)
                    points, target = points.cuda(opt.gpu_id), target.cuda(opt.gpu_id)
                    optimizer.zero_grad()
                    classifier = classifier.train()
                    pred, trans, trans_feat = classifier(points)
                    loss = F.nll_loss(pred, target)
                    train_nll_avg_loss += loss.cpu().detach().numpy()
                    if opt.feature_transform:
                        loss += feature_transform_regularizer(trans_feat) * 0.001
                    loss.backward()
                    optimizer.step()
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.data).cpu().sum()
                    total_train_correct += correct
                    print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                    epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

                end_time = default_timer()
                total_test_correct = 0.
                total_test_testset = 0.
                test_nll_avg_loss = 0.
                test_iter = 0
                # test for the whoel test dataset
                for i, data in tqdm(enumerate(testdataloader, 0)):
                    test_iter += 1
                    points, target = data
                    target = target[:, 0]
                    points = points.transpose(2, 1)
                    points, target = points.cuda(opt.gpu_id), target.cuda(opt.gpu_id)
                    classifier = classifier.eval()
                    pred, _, _ = classifier(points)
                    loss = F.nll_loss(pred, target)
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.data).cpu().sum()
                    total_test_correct += correct.item()
                    total_test_testset += points.size()[0]
                    test_nll_avg_loss += loss.cpu().detach().numpy()
                    print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                    epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batchSize)))

                # log information
                logger.append(
                    [epoch, end_time - start_time, train_nll_avg_loss / train_iter, total_train_correct / len(dataset),
                     test_nll_avg_loss / test_iter, total_test_correct / total_test_testset])

                # save model
                if epoch % 20 == 0:
                    print('save model checkpoints in output file path')
                    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=32,
                        help='input batch size',)
    parser.add_argument('--num_points', type=int, default=2500,
                        help='input batch size')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--nepoch', type=int, default=50,
                        help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls',
                        help='output folder')
    parser.add_argument('--model', type=str, default='',
                        help='model path')
    parser.add_argument('--dataset', type=str, default="data\modelnet40_normal_resampled",
                        help="dataset path")
    parser.add_argument('--feature_transform', action='store_true',
                        help="use feature transform")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="the index of gpu")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='the learning rate of classification model')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for adaptive optimizers')
    parser.add_argument('--train_times', type=int, default=5,
                        help='the traininig times of this classification demo')
    parser.add_argument('--fixed_lr', type=float, default=0.01,
                        help='the fixed learning rate for the fixed step optimizer method')
    parser.add_argument('--result_file', type=str, default='result',
                        help='the result file path to save')

    opt = parser.parse_args()
    print(opt)

    train()

