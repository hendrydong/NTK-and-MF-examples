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
h_dim = 10
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
# network
weight = np.load('./save/mnist_10000/net1_0.npy')
u = np.load('./save/mnist_10000/net2_0.npy')

net0 = nn.Sequential(nn.Linear(784, h_dim, bias=False).cuda(),nn.Tanh())
net0[0].weight.data = torch.tensor(weight).float().cuda()
train_loader_pre = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=1024, shuffle=True)
for x_, y_ in train_loader_pre:
    x_ = x_.view(-1, 28 * 28).cuda()
    o = net0(x_) # b x h
    break
S = np.std(o.detach().cpu().numpy(),0).reshape(1,u.shape[1])
u = u*S
P = np.sqrt(np.sum(np.squeeze(u)**2,0))
P = P/np.sum(P)

for s in range(save_iter):
    net1 = nn.Sequential(nn.Linear(784, h_dim, bias=False).cuda(),nn.Tanh())

    c = np.random.choice(np.arange(10000),size = h_dim,p = P)
    w = weight[c]
    net1[0].weight.data = torch.tensor(w).float().cuda()

    net2 = nn.Sequential(nn.Linear(h_dim,10, bias=True).cuda())
    torch.nn.init.normal_(net2[0].weight/np.sqrt(h_dim))
    net = nn.Sequential(net1,net2)
    Loss = nn.CrossEntropyLoss()
    op = optim.Adam(net2.parameters(), lr=lr)
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
        np.save('./results/mnist_s/acc_is_%d'%s,acc_full)
        np.save('./results/mnist_s/loss_is_%d'%s,loss_full)
        np.save('./results/mnist_s/u_is_%d'%s,u_full)


