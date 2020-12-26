import os
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--start_idx', default=0, type=int, help='save path')
parser.add_argument('--save_iter', default=1, type=int, help='save path')
args = parser.parse_args()


class NTK(nn.Module):
    """NTK: Linearization of NN"""
    def __init__(self, h_dim):
        super(NTK, self).__init__()
        self.h_dim = h_dim
        self.net0 = nn.Linear(784, h_dim, bias=False).cuda()
        torch.nn.init.normal_(self.net0.weight,mean=0.0, std=1.0)
        self.net_nt = nn.Linear(784, 10*h_dim, bias=True).cuda()
        torch.nn.init.normal_(self.net_nt.weight,mean=0.0, std=1e-5)
        self.net_rf = nn.Linear(h_dim, 10, bias=True).cuda()
        torch.nn.init.normal_(self.net_rf.weight,mean=0.0, std=1.0/np.sqrt(h_dim))
        self.out = lambda x:x
        self.train_parameters = nn.ModuleList([self.net_nt,self.net_rf])
    def forward(self, x):
        feat0 = self.net0(x)
        feat = 4/(torch.exp(feat0)+torch.exp(-feat0))**2 # n x h
        f_nt = self.net_nt(x) # n x 10 h 
        f_nt = f_nt.view(-1,self.h_dim,10)
        feat = feat.view(-1,self.h_dim,1)
        o1 = torch.sum(f_nt*feat,1)
        o2 = self.net_rf(feat0)
        o = self.out(o1+o2)
        return o


# training parameters
batch_size = 128
lr = 1e-5
train_epoch = 50
h_dim = 1000
save = False
save_iter = args.save_iter
# data_loader

transform = transforms.Compose([
        transforms.ToTensor()
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)
# network
for s in range(save_iter):
    

    net = NTK(h_dim).cuda()
    op = optim.Adam(net.train_parameters.parameters(), lr=lr)
    # Cross Entropy loss
    Loss = nn.CrossEntropyLoss()
    acc_full = []
    loss_full = []
    for epoch in range(train_epoch):
        loss_train = []
        for x_, y_ in train_loader:
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)
            #print(y_pred.shape)
            loss = Loss(y_pred,(y_))
            op.zero_grad()
            loss.backward()
            op.step()
            loss_train+=[loss.item()]

        acc = []
        loss_test = []
        for x_, y_ in test_loader:
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)
            loss = Loss(y_pred,(y_))
            loss_test+=[loss.item()]
            acc += [torch.argmax(y_pred,1)==y_]
        acc = torch.mean(torch.cat(acc).float())
        acc_full.append(acc.item())
        loss_full.append((np.mean(loss_train[-10:]),np.mean(loss_test)))
        print(s,epoch,loss.item(),loss_full[-1],acc.item())

    np.save('./results/loss_ntk_%d_%d'%(h_dim,s),loss_full)
    np.save('./results/acc_ntk__%d_%d'%(h_dim,s),acc_full)

