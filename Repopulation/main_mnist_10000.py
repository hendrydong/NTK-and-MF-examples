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
lr = 1e-3
train_epoch = 20
h_dim = 10000
save = True
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
    net2 = nn.Sequential(nn.Linear(h_dim,10, bias=True).cuda())
    torch.nn.init.normal_(net2[0].weight,mean = 0,std = 1/np.sqrt(h_dim))
    net = nn.Sequential(net1,net2)
    Loss = nn.CrossEntropyLoss()
    op = optim.Adam(net.parameters(), lr=lr,weight_decay = 1e-3)
    acc_full = []
    loss_full = []
    u_full = []
    for epoch in range(train_epoch):
        loss_train = []
        net.train()
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
        net.eval()
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
        np.save('./save/mnist_10000/net1_%d'%s,np.array(net1[0].weight.cpu().detach().numpy()))
        np.save('./save/mnist_10000/net2_%d'%s,np.array(net2[0].weight.cpu().detach().numpy()))



