#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn.functional as F
import argparse


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
                        
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
                        
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of episode to train ')
                        
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='Use SGD')
                        
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
                        
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
                        
    parser.add_argument('--scheduler', type=str, metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
                        
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
                        
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
                        
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
                        
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
                        
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
                        
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
                        
    parser.add_argument('--num_class', type=int, default=4, metavar='num_class',
                        help='Number of classes')
                        
    parser.add_argument('--data', '--dataset', help='Path to data', required=True, default=None)

    parser.add_argument('--metric',type=str,default='IOU', help='Type of metric IOU or Dice')

    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
                        
    args = parser.parse_args()
    
    return args


def cal_loss(pred, gold, class_weights, smoothing):

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb * class_weights).sum(dim=1).mean()		
    else:	
        loss = F.cross_entropy(pred, gold, weight=class_weights, reduction='mean')

    return loss


def calculate_metric(pred_np, seg_np, num_class):
    I_all = np.zeros(num_class)
    U_all = np.zeros(num_class)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(num_class):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all
    
def calculate_metric_dice(pred_np, seg_np, num_class, smoothing=1e-5):

    D_all = np.zeros(num_class)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(num_class):
            TP = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            FP = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] != sem))
            FN = np.sum(np.logical_and(pred_np[sem_idx] != sem, seg_np[sem_idx] == sem))
            D = (2 * TP + smoothing) / (2 * TP + FP + FN + smoothing)
            D_all[sem] += D
    return D_all / seg_np.shape[0]


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

        #why not logging
        
def angle_axis(angle, axis):
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )

    return R.float()


class PCRotate(object):
    def __init__(self, axis=np.array([0.0, 0.0, 1.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        pc_xyz = points[:, 0:3]
        points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())

        return points


class PCToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()
