"""
# !/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import load_dataset.data_loader as data_loader
import os
import math
import copy
from datetime import datetime
import multiprocessing
from utils import Utils
from template.drop import drop_path


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hswish()'

class ECALayer(nn.Module):
    def __init__(self, in_channels, out_channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(in_channels, 2)+b)/gamma))
        k = t if t%2 else t+1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1,1,kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, expansion_rate=3, act_func='h_swish', drop_connect_rate=0.0):
        super(BasicBlock, self).__init__()
        interChannels = expansion_rate*planes

        self.conv1 = nn.Conv2d(in_planes, interChannels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.sep_conv = nn.Conv2d(interChannels, interChannels, kernel_size=kernel_size, stride=stride, padding=int((kernel_size-1)/2), bias=False,groups=interChannels)
        self.bn2 = nn.BatchNorm2d(interChannels)

        self.point_conv = nn.Conv2d(interChannels, planes, kernel_size=1, stride=stride, padding=0, bias=False,groups=1)
        self.bn3 = nn.BatchNorm2d(planes)
        if act_func == 'relu':
            self.act_func = nn.ReLU(inplace=True)
            self.se = nn.Sequential()
        else:
            self.act_func = Hswish(inplace=True)
            self.se = ECALayer(interChannels, interChannels)
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x):
        out = self.act_func(self.bn1(self.conv1(x)))
        out = self.act_func(self.bn2(self.sep_conv(out)))
        out = self.se(out)
        out = self.bn3(self.point_conv(out))
        if self.drop_connect_rate > 0.:
            out = drop_path(out, drop_prob=self.drop_connect_rate, training=self.training)
        return out


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        self.Hswish = Hswish(inplace=True)
        #generated_init


    def forward(self, x):
        #generate_forward


        out = out.view(out.size(0), -1)
        out = self.Hswish(self.dropout(self.linear1(out)))
        out = self.linear(out)
        return out


class TrainModel(object):
    def __init__(self, is_test, batch_size, weight_decay):
        if is_test:
            full_trainloader = data_loader.get_train_loader('../datasets/CIFAR10_data', batch_size=batch_size, augment=True,shuffle=True, random_seed=2312391, show_sample=False,num_workers=4, pin_memory=True)
            testloader = data_loader.get_test_loader('../datasets/CIFAR10_data', batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)
            self.full_trainloader = full_trainloader
            self.testloader = testloader
        else:
            trainloader, validate_loader = data_loader.get_train_valid_loader('../datasets/CIFAR10_data', batch_size=256,augment=True, subset_size=1,valid_size=0.1, shuffle=True,random_seed=2312390, show_sample=False,num_workers=4, pin_memory=True)
            self.trainloader = trainloader
            self.validate_loader = validate_loader

        net = EvoCNNModel()
        cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss()
        net = net.cuda()
        best_acc = 0.0
        self.net = net
        self.criterion = criterion
        self.best_acc = best_acc
        self.best_epoch = 0
        self.file_id = os.path.basename(__file__).split('.')[0]
        self.weight_decay = weight_decay

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def train(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for ii, data in enumerate(self.trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

            if epoch==0 and ii==0:
                params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
                self.log_record('#parameters:%d' % (params))
        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f'% (epoch+1, running_loss/total, (correct/total)))

    def final_train(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for ii, data in enumerate(self.full_trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

            if epoch==0 and ii==0:
                params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
                self.log_record('#parameters:%d' % (params))
        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f' % (epoch + 1, running_loss / total, (correct / total)))

    def validate(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        is_terminate = 0
        for _, data in enumerate(self.validate_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if epoch >= self.best_epoch + 4 or correct / total - self.best_acc < -0.03:
            is_terminate = 1
        if correct / total > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = correct / total
        self.log_record('Validate-Epoch:%4d,  Validate-Loss:%.4f, Acc:%.4f'%(epoch + 1, test_loss/total, correct/total))
        return is_terminate

    def process(self):
        total_epoch = Utils.get_params('network', 'epoch_test')
        min_epoch_eval = Utils.get_params('network', 'min_epoch_eval')

        lr_rate = 0.08
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr_rate, momentum=0.9, weight_decay=4e-5, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, min_epoch_eval)
        
        is_terminate = 0
        # params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        # self.log_record('#parameters:%d' % (params))

        for p in range(total_epoch):
            if not is_terminate:
                self.train(p, optimizer)
                scheduler.step()
                is_terminate = self.validate(p)
            else:
                return self.best_acc
        return self.best_acc

    def process_test(self):
        # params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        # self.log_record('#parameters:%d' % (params))
        total_epoch = Utils.get_params('network', 'epoch_test')
        lr_rate = 0.08
        optimizer = optim.SGD(self.net.parameters(), lr=lr_rate, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)

        for p in range(total_epoch):
            self.final_train(p, optimizer)
            self.test(p)
            scheduler.step()
        return self.best_acc

    def test(self,p):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.testloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if correct / total > self.best_acc:
            self.best_acc = correct / total
        self.log_record('Test-Loss:%.4f, Acc:%.4f' % (test_loss / total, correct / total))

class RunModel(object):
    def do_work(self, gpu_id, curr_gen, file_id, is_test, batch_size=None, weight_decay=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        try:
            m = TrainModel(is_test, batch_size, weight_decay)
            m.log_record('Used GPU#%s, worker name:%s[%d]'%(gpu_id, multiprocessing.current_process().name, os.getpid()), first_time=True)
            if is_test:
                best_acc = m.process_test()
            else:
                best_acc = m.process()

        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
        finally:
            m.log_record('Finished-Acc:%.4f'%best_acc)

            f = open('./populations/acc_%02d.txt'%(curr_gen), 'a+')
            f.write('%s=%.5f\n'%(file_id, best_acc))
            f.flush()
            f.close()
"""


