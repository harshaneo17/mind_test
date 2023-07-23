#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
import numpy as np
import sklearn.metrics as metrics
from time import time, ctime
from torch.utils.data import DataLoader

from data.data import Dataset
from models.model import Model
from utils.utils import cal_loss, IOStream, calculate_metric, get_args, calculate_metric_dice


class PointCloudSegmentation:
    def __init__(self):
        self.args = None
        self.io = None
        self.train_loader = None
        self.validation_loader = None
        self.device = None
        self.model = None
        self.opt = None
        self.scheduler = None
        self.best_val_iou = 0
        self.metric = None

    def file_check(self):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists('checkpoints/' + self.args.exp_name):
            os.makedirs('checkpoints/' + self.args.exp_name)
        if not os.path.exists('checkpoints/' + self.args.exp_name + '/' + 'models'):
            os.makedirs('checkpoints/' + self.args.exp_name + '/' + 'models')

    def train_one_epoch(self, epoch):
        time_str = 'Train start:' + ctime(time())
        self.io.cprint(time_str)
        train_loss = 0.0
        count = 0.0
        self.model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, seg, class_weights in self.train_loader:
            data, seg, class_weights = data.to(self.device, dtype=torch.float), seg.to(self.device), class_weights[
                0].to(self.device)
            data = data.permute(0, 2, 1)

            batch_size = data.size()[0]
            self.opt.zero_grad()
            seg_pred = self.model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = cal_loss(seg_pred.view(-1, self.args.num_class), seg.view(-1, 1).squeeze(), class_weights,
                            smoothing=True)
            loss.backward()
            self.opt.step()
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            train_true_cls.append(seg_np.reshape(-1))
            train_pred_cls.append(pred_np.reshape(-1))
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        if self.args.scheduler == 'cos':
            self.scheduler.step()
        elif self.args.scheduler == 'step':
            if self.opt.param_groups[0]['lr'] > 1e-5:
                self.scheduler.step()
            if self.opt.param_groups[0]['lr'] < 1e-5:
                for param_group in self.opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls) #if train_true_cls else np.array([])
        train_pred_cls = np.concatenate(train_pred_cls) #if train_pred_cls else np.array([])
        train_true_seg = np.concatenate(train_true_seg, axis=0) #if train_true_seg else np.array([])
        train_pred_seg = np.concatenate(train_pred_seg, axis=0) #if train_pred_seg else np.array([])
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        if self.args.metric == 'IOU':
            train_ious = calculate_metric(train_pred_seg, train_true_seg, self.args.num_class)
        else:
            train_ious = calculate_metric_dice(train_pred_seg, train_true_seg, self.args.num_class, smoothing=1e-5)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (
            epoch,
            train_loss * 1.0 / count,
            train_acc,
            avg_per_class_acc,
            np.mean(train_ious))
        self.io.cprint(outstr)
        torch.save(self.model.state_dict(), 'checkpoints/%s/models/latest_model.t7' % self.args.exp_name)
        time_str = 'Train complete:' + ctime(time())
        self.io.cprint(time_str)

    def eval_one_epoch(self, epoch):
        val_loss = 0.0
        count = 0.0
        self.model.eval()
        val_true_cls = []
        val_pred_cls = []
        val_true_seg = []
        val_pred_seg = []
        for data, seg, class_weights in self.validation_loader:
            data, seg, class_weights = data.to(self.device, dtype=torch.float), seg.to(self.device), class_weights[
                0].to(self.device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = self.model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = cal_loss(seg_pred.view(-1, self.args.num_class), seg.view(-1, 1).squeeze(), class_weights,
                            smoothing=False)
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            val_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            val_true_cls.append(seg_np.reshape(-1))
            val_pred_cls.append(pred_np.reshape(-1))
            val_true_seg.append(seg_np)
            val_pred_seg.append(pred_np)
        val_true_cls = np.concatenate(val_true_cls)
        val_pred_cls = np.concatenate(val_pred_cls)
        val_acc = metrics.accuracy_score(val_true_cls, val_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(val_true_cls, val_pred_cls)
        val_true_seg = np.concatenate(val_true_seg, axis=0)
        val_pred_seg = np.concatenate(val_pred_seg, axis=0)
        if self.args.metric == 'IOU':
            val_ious = calculate_metric(val_pred_seg, val_true_seg, self.args.num_class)
        else:
            val_ious = calculate_metric_dice(val_pred_seg, val_true_seg, self.args.num_class, smoothing=1e-5)
        outstr = 'Validation %d, loss: %.6f, val acc: %.6f, val avg acc: %.6f, val iou: %.6f' % (
            epoch,
            val_loss * 1.0 / count,
            val_acc,
            avg_per_class_acc,
            np.mean(val_ious))
        self.io.cprint(outstr)
        return val_ious

    def run(self):
        self.args = get_args()
        self.file_check()

        self.io = IOStream('checkpoints/' + self.args.exp_name + '/run.log')
        self.io.cprint('Program start: %s' % ctime(time()))

        

        self.train_loader = DataLoader(
            Dataset(DATA_DIR=self.args.data, num_pts=self.args.num_points, partition="train", tile_size=30),
            num_workers=8, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.validation_loader = DataLoader(
            Dataset(DATA_DIR=self.args.data, num_pts=self.args.num_points, partition="validation", tile_size=30),
            num_workers=8, batch_size=self.args.batch_size, shuffle=True, drop_last=False)

        self.device = torch.device("cpu")

        self.model = Model(self.args.k, self.args.emb_dims, self.args.dropout).to(self.device)

        self.model = nn.DataParallel(self.model)

        if self.args.optimizer == 'SGD':
            print("Use SGD")
            self.opt = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                 weight_decay=1e-4)
        elif self.args.optimizer == 'Adam':
            print("Use Adam")
            self.opt = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        elif self.args.optimizer == 'AdamW':
            print("Use AdamW")
            self.opt = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        if self.args.scheduler == 'cos':
            self.scheduler = CosineAnnealingLR(self.opt, self.args.epochs, eta_min=1e-3)
        elif self.args.scheduler == 'step':
            self.scheduler = ExponentialLR(self.opt, 0.7)

        for epoch in range(self.args.epochs):
            self.io.cprint('---------------------Epoch %d/%d---------------------' % (epoch, self.args.epochs))
            self.train_one_epoch(epoch)

            if epoch % 5 == 0:
                val_ious = self.eval_one_epoch(epoch)

                if np.mean(val_ious) >= self.best_val_iou:
                    self.best_val_iou = np.mean(val_ious)
                    torch.save(self.model.state_dict(),
                               'checkpoints/%s/models/best_model.t7' % self.args.exp_name)
                    self.io.cprint(
                        'Best model Epoch_%d: checkpoints/%s/models/best_model.t7' % (epoch, self.args.exp_name))
                time_str = 'Validation complete:' + ctime(time())
                self.io.cprint(time_str)

        if self.args.eval == True:
            self.run_evaluation()

    def evaluate_model(self):
        self.io.cprint('Loading the best model for evaluation...')
        self.model.load_state_dict(torch.load(f'checkpoints/{self.args.exp_name}/models/best_model.t7'))
        self.model.eval()

        self.io.cprint('Evaluating the model on the validation data...')
        val_true_cls = []
        val_pred_cls = []
        val_true_seg = []
        val_pred_seg = []

        with torch.no_grad():
            for data, seg, class_weights in self.validation_loader:
                data = data.to(self.device, dtype=torch.float)
                data = data.permute(0, 2, 1)
                seg_pred = self.model(data)
                seg_pred = seg_pred.permute(0, 2, 1)

                pred = seg_pred.max(dim=2)[1]
                seg_np = seg.cpu().numpy()
                pred_np = pred.cpu().numpy()

                val_true_cls.append(seg_np.reshape(-1))
                val_pred_cls.append(pred_np.reshape(-1))
                val_true_seg.append(seg_np)
                val_pred_seg.append(pred_np)

        val_true_cls = np.concatenate(val_true_cls)
        val_pred_cls = np.concatenate(val_pred_cls)
        val_acc = metrics.accuracy_score(val_true_cls, val_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(val_true_cls, val_pred_cls)
        val_true_seg = np.concatenate(val_true_seg, axis=0)
        val_pred_seg = np.concatenate(val_pred_seg, axis=0)
        if self.args.metric == 'IOU':
            val_ious = calculate_metric(val_pred_seg, val_true_seg, self.args.num_class)
        else:
            val_ious = calculate_metric_dice(val_pred_seg, val_true_seg, self.args.num_class, smoothing=1e-5)

        self.io.cprint('Validation Results:')
        self.io.cprint(f'Accuracy: {val_acc:.6f}')
        self.io.cprint(f'Average Per-Class Accuracy: {avg_per_class_acc:.6f}')
        self.io.cprint(f'IOU: {np.mean(val_ious):.6f}')

    def run_evaluation(self):
        self.file_check()
        self.io = IOStream('checkpoints/' + self.args.exp_name + '/run.log')
        self.io.cprint('Program start: %s' % ctime(time()))

        self.validation_loader = DataLoader(
            Dataset(DATA_DIR=self.args.data, num_pts=self.args.num_points, partition="validation", tile_size=30),
            num_workers=8, batch_size=self.args.batch_size, shuffle=True, drop_last=False)

        self.device = torch.device("cpu")

        self.model = Model(self.args.k, self.args.emb_dims, self.args.dropout).to(self.device)

        self.model = nn.DataParallel(self.model)

        self.evaluate_model()

        self.io.cprint('Evaluation complete:' + ctime(time()))


if __name__ == "__main__":
    pcd_segmentation = PointCloudSegmentation()
    pcd_segmentation.run()



