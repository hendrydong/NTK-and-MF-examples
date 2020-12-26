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


# training parameters
batch_size = 128
lr = 1e-4
train_epoch = 50
h_dim = 1000
save = False
save_iter = 1
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
for s in range(save_iter):
    net1 = nn.Sequential(nn.Linear(784, h_dim, bias=False).cuda(),nn.Tanh())
    torch.nn.init.normal_(net1[0].weight)
    if save:
        np.save('%d_init'%s,list(net1.parameters())[0].detach().cpu().numpy())
    net2 = nn.Sequential(nn.Linear(h_dim,10, bias=False).cuda())
    net = nn.Sequential(net1,net2)
    op = optim.Adam(net2.parameters(), lr=lr)
    # Cross Entropy loss
    Loss = nn.CrossEntropyLoss()
    acc_full = []
    loss_full = []
    u_full = []
    for epoch in range(train_epoch):
        loss_train = []
        for x_, y_ in train_loader:
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)
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
        
        u_full.append(np.array(net2[0].weight.cpu().detach().numpy()))
    if save:
        print(torch.sum(net2[0].weight.data**2))
        np.save('./u/u_random_%d'%s,u_full)

    np.save('./results/loss_random_%d_%d'%(h_dim,s),loss_full)
    np.save('./results/acc_random_%d_%d'%(h_dim,s),acc_full)


