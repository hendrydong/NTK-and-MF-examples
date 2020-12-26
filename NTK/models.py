import torch
from torch import optim, nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from ntk import ntk_v3


# Standard strategy: eta_theta = eta_u * alpha/m
def train_standard(train_loader, test_loader, h_dim, alpha, train_epoch, lr = 1, m = 0, SEED = 2020, print_result=True):
    
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    loss_full = []
    acc_full = []
    net1 = nn.Sequential(nn.Linear(784, h_dim, bias=False).cuda(),nn.Tanh())
    torch.nn.init.normal_(net1[0].weight,mean=0.0, std=1.0)
    net2 = nn.Linear(h_dim,10, bias=False).cuda()
    torch.nn.init.normal_(net2.weight,mean=0.0, std=1.0*alpha/h_dim)
    theta0 = list(net1.parameters())[0].detach().cpu().numpy().copy()
    u0 =  list(net2.parameters())[0].detach().cpu().numpy().copy()
    net = nn.Sequential(net1,net2)
    Loss = nn.CrossEntropyLoss()

    coeff = 1
    eta = lr
    op = optim.SGD(net.parameters(), lr = eta, momentum = m)
    relative_change = []
    for epoch in range(train_epoch):
        loss_train = []
        for x_, y_ in train_loader:
            # train discriminator D
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)*coeff
            #print(y_pred.shape)
            loss = Loss(y_pred,y_)
            op.zero_grad()
            loss.backward()
            op.step()
            loss_train+=[loss.item()]
        acc = []
        loss_test = []
        for x_, y_ in test_loader:
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)*coeff
            loss = Loss(y_pred,y_)
            loss_test+=[loss.item()]
            acc += [torch.argmax(y_pred,1)==y_]
        acc = torch.mean(torch.cat(acc).float())
        acc_full.append(acc.item())
        loss_full.append((np.mean(loss_train[-10:]),np.mean(loss_test)))
        
        theta1 = list(net1.parameters())[0].detach().cpu().numpy().copy()
        u1 = list(net2.parameters())[0].detach().cpu().numpy().copy()
        dtheta = (np.mean(np.sum((theta1-theta0)**2,1)**0.5))
        du = (np.mean(np.sum((u1-u0)**2,0)**0.5))/alpha*h_dim
        if print_result:
            print('epoch %d'%epoch,'loss (train,test):%.2e;%.2e'%loss_full[-1],'acc:%.6f'%acc.item())
            print('dtheta:',dtheta)
            print('du:',du)
        relative_change.append((dtheta,du))

    return relative_change,acc_full,loss_full

# Standard strategy: eta_theta = eta_u * alpha/m
def train_standard_2(train_loader, test_loader, h_dim, alpha, train_epoch, lr = 1, m = 0, SEED = 2020, print_result=True):
    
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    loss_full = []
    acc_full = []
    net1 = nn.Sequential(nn.Linear(784, h_dim, bias=False).cuda(),nn.Tanh())
    torch.nn.init.normal_(net1[0].weight,mean=0.0, std=1.0/np.sqrt(h_dim))
    net2 = nn.Linear(h_dim,10, bias=False).cuda()
    torch.nn.init.normal_(net2.weight,mean=0.0, std=1.0*alpha/h_dim)
    theta0 = list(net1.parameters())[0].detach().cpu().numpy().copy()
    u0 =  list(net2.parameters())[0].detach().cpu().numpy().copy()
    net = nn.Sequential(net1,net2)
    Loss = nn.CrossEntropyLoss()

    coeff = 1
    eta = lr
    op = optim.SGD(net.parameters(), lr = eta, momentum = m)
    relative_change = []
    for epoch in range(train_epoch):
        loss_train = []
        for x_, y_ in train_loader:
            # train discriminator D
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)*coeff
            #print(y_pred.shape)
            loss = Loss(y_pred,y_)
            op.zero_grad()
            loss.backward()
            op.step()
            loss_train+=[loss.item()]
        acc = []
        loss_test = []
        for x_, y_ in test_loader:
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)*coeff
            loss = Loss(y_pred,y_)
            loss_test+=[loss.item()]
            acc += [torch.argmax(y_pred,1)==y_]
        acc = torch.mean(torch.cat(acc).float())
        acc_full.append(acc.item())
        loss_full.append((np.mean(loss_train[-10:]),np.mean(loss_test)))
        
        theta1 = list(net1.parameters())[0].detach().cpu().numpy().copy()
        u1 = list(net2.parameters())[0].detach().cpu().numpy().copy()
        dtheta = (np.mean(np.sum((theta1-theta0)**2,1)**0.5))
        du = (np.mean(np.sum((u1-u0)**2,0)**0.5))/alpha*h_dim
        if print_result:
            print('epoch %d'%epoch,'loss (train,test):%.2e;%.2e'%loss_full[-1],'acc:%.6f'%acc.item())
            print('dtheta:',dtheta)
            print('du:',du)
        relative_change.append((dtheta,du))

    return relative_change,acc_full,loss_full




# Theoretical strategy (1): eta_theta = eta_u
def train(train_loader,test_loader,h_dim,alpha,train_epoch,lr = 1, m = 0, SEED = 2020,print_result=True):
    print('Seed:',SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    loss_full = []
    acc_full = []
    net1 = nn.Sequential(nn.Linear(784, h_dim, bias=False).cuda(),nn.Tanh())
    torch.nn.init.normal_(net1[0].weight,mean=0.0, std=1.0)
    net2 = nn.Linear(h_dim,10, bias=False).cuda()
    torch.nn.init.normal_(net2.weight,mean=0.0, std=1.0)
    theta0 = list(net1.parameters())[0].detach().cpu().numpy().copy()
    u0 =  list(net2.parameters())[0].detach().cpu().numpy().copy()
    net = nn.Sequential(net1,net2)
    Loss = nn.CrossEntropyLoss()
    coeff = alpha / h_dim
    eta = h_dim / alpha * lr
    op = optim.SGD(net.parameters(),lr = eta, momentum = m)
    relative_change = []
    for epoch in range(train_epoch):
        loss_train = []
        for x_, y_ in train_loader:
            # train discriminator D
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)*coeff
            #print(y_pred.shape)
            loss = Loss(y_pred,y_)
            op.zero_grad()
            loss.backward()
            op.step()
            loss_train+=[loss.item()]
        acc = []
        loss_test = []
        for x_, y_ in test_loader:
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)*coeff
            loss = Loss(y_pred,y_)
            loss_test+=[loss.item()]
            acc += [torch.argmax(y_pred,1)==y_]
        acc = torch.mean(torch.cat(acc).float())
        acc_full.append(acc.item())
        loss_full.append((np.mean(loss_train[-10:]),np.mean(loss_test)))
        #print(epoch,loss.item(),loss_full[-1],acc.item())
        theta1 = list(net1.parameters())[0].detach().cpu().numpy().copy()
        u1 = list(net2.parameters())[0].detach().cpu().numpy().copy()
        dtheta = np.mean(np.sum((theta1-theta0)**2,1)**0.5)
        du = (np.mean(np.sum((u1-u0)**2,0)**0.5))
        if print_result:
            print('epoch %d'%epoch,'loss (train,test):%.2e;%.2e'%loss_full[-1],'acc:%.6f'%acc.item())
            print('dtheta:',dtheta)
            print('du:',du)
        relative_change.append((dtheta,du))

    return relative_change,acc_full,loss_full



# Theoretical strategy (2): eta_theta = eta_u * alpha
def train_2(train_loader,test_loader, h_dim,alpha,train_epoch, lr = 1,m = 0, SEED = 2020,print_result=True):
    
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    loss_full = []
    acc_full = []
    coeff = alpha/h_dim
    net1 = nn.Sequential(nn.Linear(784, h_dim, bias=False).cuda(),nn.Tanh())
    torch.nn.init.normal_(net1[0].weight,mean=0.0, std=1.0)
    net2 = nn.Linear(h_dim,10, bias=False).cuda()
    torch.nn.init.normal_(net2.weight,mean=0.0, std=1.0*alpha)
    #torch.nn.init.normal_(net2.bias,mean=0.0, std=1.0*alpha)
    theta0 = list(net1.parameters())[0].detach().cpu().numpy().copy()
    u0 =  list(net2.parameters())[0].detach().cpu().numpy().copy()
    net = nn.Sequential(net1,net2)
    Loss = nn.CrossEntropyLoss()

    eta = lr*h_dim
    op = optim.SGD(net.parameters(),lr = eta, momentum = m)
    relative_change = []
    for epoch in range(train_epoch):
        loss_train = []
        for x_, y_ in train_loader:
            # train discriminator D
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)/h_dim
            #print(y_pred.shape)
            loss = Loss(y_pred,y_)
            op.zero_grad()
            loss.backward()
            op.step()
            loss_train+=[loss.item()]
        acc = []
        loss_test = []
        for x_, y_ in test_loader:
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)/h_dim
            loss = Loss(y_pred,y_)
            loss_test+=[loss.item()]
            acc += [torch.argmax(y_pred,1)==y_]
        acc = torch.mean(torch.cat(acc).float())
        acc_full.append(acc.item())
        loss_full.append((np.mean(loss_train[-10:]),np.mean(loss_test)))
        #print(epoch,loss.item(),loss_full[-1],acc.item())
        theta1 = list(net1.parameters())[0].detach().cpu().numpy().copy()
        u1 = list(net2.parameters())[0].detach().cpu().numpy().copy()
        
        dtheta = (np.mean(np.sum((theta1-theta0)**2,1)**0.5))
        du = (np.mean(np.sum((u1-u0)**2,0)**0.5))/alpha
        if print_result:
            print('epoch %d'%epoch,'loss (train,test):%.2e;%.2e'%loss_full[-1],'acc:%.6f'%acc.item())
            print('dtheta:',dtheta)
            print('du:',du)
        relative_change.append((dtheta,du))

    return relative_change,acc_full,loss_full

# Save Linearization during the training process
def train_ntk(train_loader, test_loader, h_dim, alpha, train_epoch, lr = 1, m = 0, SEED = 2020,print_result=True):
    
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
    loss_full = []
    acc_full = []
    net1 = nn.Sequential(nn.Linear(28*28, h_dim, bias=False).cuda(),nn.Tanh())
    torch.nn.init.normal_(net1[0].weight)
    net2 = nn.Linear(h_dim,10, bias=False).cuda()
    torch.nn.init.normal_(net2.weight,mean=0.0, std=1.0*alpha/h_dim)
    #torch.nn.init.normal_(net2.bias,mean=0.0, std=1.0*alpha/h_dim)
    theta0 = list(net1.parameters())[0].detach().cpu().numpy().copy()
    u0 =  list(net2.parameters())[0].detach().cpu().numpy().copy()
    net = nn.Sequential(net1,net2)
    
    net1_init = copy.deepcopy(net1)
    net2_init = copy.deepcopy(net2)
    net_init = nn.Sequential(net1_init,net2_init)
    Loss = nn.CrossEntropyLoss()

    coeff = 1#alpha/h_dim
    eta = lr
    op = optim.SGD(net.parameters(),lr = eta,momentum=m)
    relative_change = []
    for epoch in range(train_epoch):
        #if epoch==int(0.3*train_epoch) and lr >0.1:
        #    lr = 0.1
        #    op = optim.SGD(net.parameters(),lr = 0.1,momentum=m)
        #if epoch==int(0.6*train_epoch) and lr >0.01:
        #    op = optim.SGD(net.parameters(),lr = 0.01,momentum=m)
        loss_train = []
        loss_train_ntk = []
        for x_, y_ in train_loader:
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)
            #y_pred = net(X_tr)
            #print(y_pred.shape)
            #print(y_pred.dtype)
            #print((Y_tr).dtype)
            loss = Loss(y_pred,y_)
            op.zero_grad()
            loss.backward()
            op.step()
            
        j = 0
        for x_, y_ in train_loader:
            if j==10:
                break
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)
            loss = Loss(y_pred,y_)
            y_tr_ntk = ntk_v3(net,net_init,x_)
            loss_ntk_tr = Loss(y_tr_ntk,y_)
            loss_train+=[loss.item()]
            loss_train_ntk+=[loss_ntk_tr.item()]
            j+=1
        acc = []
        loss_test = []
        #for x_, y_ in test_loader:
        #x_ = x_.view(-1, 28 * 28).cuda()
        #y_ = y_.cuda()
        loss_ntk = []
        acc_ntk = []
        for x_, y_ in test_loader:
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)
            loss = Loss(y_pred,y_)
            loss_test+=[loss.item()]
            y_te_ntk = ntk_v3(net,net_init,x_)
            loss_ntk_te = Loss(y_te_ntk,y_)
            loss_ntk += [loss_ntk_te.item()]
            
            acc += [torch.argmax(y_pred,1)==y_]
            acc_ntk +=[torch.argmax(y_te_ntk,1)==y_]
        loss_ntk_te = np.mean(loss_ntk)
        acc_ntk_mean = torch.mean(torch.cat(acc_ntk).float())
        acc = torch.mean(torch.cat(acc).float())
        acc_full.append((acc.item(),acc_ntk_mean.item()))
        loss_full.append((np.mean(loss_train),np.mean(loss_test),np.mean(loss_train_ntk),loss_ntk_te))
        
        
        #print(epoch,loss.item(),loss_full[-1],acc.item())
        theta1 = list(net1.parameters())[0].detach().cpu().numpy().copy()
        u1 = list(net2.parameters())[0].detach().cpu().numpy().copy()
        dtheta = (np.mean(np.sum((theta1-theta0)**2,1)**0.5))
        du = (np.mean(np.sum((u1-u0)**2,0)**0.5))/alpha*h_dim
        if print_result:
        
            print('epoch %d'%epoch,'loss (train,test):%.2e;%.2e'%loss_full[-1][:2],'acc:%.6f'%acc.item())
            print('dtheta:',dtheta)
            print('du:',du)
        relative_change.append((dtheta,du))

        #print(epoch,loss.item(),loss_full[-1],acc.item())
        #print('ntk',loss_ntk_tr.item(),loss_ntk_te,acc_ntk_mean.item())

    return relative_change,acc_full,loss_full

# Save Linearization during the training process 
def train_ntk_theta(train_loader, test_loader, h_dim, alpha, train_epoch, lr = 1, m = 0, SEED = 2020,print_result=True):
    
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
    loss_full = []
    acc_full = []
    net1 = nn.Sequential(nn.Linear(28*28, h_dim, bias=False).cuda(),nn.Tanh())
    torch.nn.init.normal_(net1[0].weight,mean=0.0, std=1.0/28)
    net2 = nn.Linear(h_dim,10, bias=False).cuda()
    torch.nn.init.normal_(net2.weight,mean=0.0, std=1.0*alpha/h_dim)
    #torch.nn.init.normal_(net2.bias,mean=0.0, std=1.0*alpha/h_dim)
    theta0 = list(net1.parameters())[0].detach().cpu().numpy().copy()
    u0 =  list(net2.parameters())[0].detach().cpu().numpy().copy()
    net = nn.Sequential(net1,net2)
    
    net1_init = copy.deepcopy(net1)
    net2_init = copy.deepcopy(net2)
    net_init = nn.Sequential(net1_init,net2_init)
    Loss = nn.CrossEntropyLoss()

    coeff = 1#alpha/h_dim
    eta = lr
    op = optim.SGD(net.parameters(),lr = eta,momentum=m)
    relative_change = []
    for epoch in range(train_epoch):
        #if epoch==int(0.3*train_epoch) and lr >0.1:
        #    lr = 0.1
        #    op = optim.SGD(net.parameters(),lr = 0.1,momentum=m)
        #if epoch==int(0.6*train_epoch) and lr >0.01:
        #    op = optim.SGD(net.parameters(),lr = 0.01,momentum=m)
        loss_train = []
        loss_train_ntk = []
        for x_, y_ in train_loader:
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)
            #y_pred = net(X_tr)
            #print(y_pred.shape)
            #print(y_pred.dtype)
            #print((Y_tr).dtype)
            loss = Loss(y_pred,y_)
            op.zero_grad()
            loss.backward()
            op.step()
            
        j = 0
        for x_, y_ in train_loader:
            if j==10:
                break
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)
            loss = Loss(y_pred,y_)
            y_tr_ntk = ntk_v3(net,net_init,x_)
            loss_ntk_tr = Loss(y_tr_ntk,y_)
            loss_train+=[loss.item()]
            loss_train_ntk+=[loss_ntk_tr.item()]
            j+=1
        acc = []
        loss_test = []
        #for x_, y_ in test_loader:
        #x_ = x_.view(-1, 28 * 28).cuda()
        #y_ = y_.cuda()
        loss_ntk = []
        acc_ntk = []
        for x_, y_ in test_loader:
            x_ = x_.view(-1, 28 * 28).cuda()
            y_ = y_.cuda()
            y_pred = net(x_)
            loss = Loss(y_pred,y_)
            loss_test+=[loss.item()]
            y_te_ntk = ntk_v3(net,net_init,x_)
            loss_ntk_te = Loss(y_te_ntk,y_)
            loss_ntk += [loss_ntk_te.item()]
            
            acc += [torch.argmax(y_pred,1)==y_]
            acc_ntk +=[torch.argmax(y_te_ntk,1)==y_]
        loss_ntk_te = np.mean(loss_ntk)
        acc_ntk_mean = torch.mean(torch.cat(acc_ntk).float())
        acc = torch.mean(torch.cat(acc).float())
        acc_full.append((acc.item(),acc_ntk_mean.item()))
        loss_full.append((np.mean(loss_train),np.mean(loss_test),np.mean(loss_train_ntk),loss_ntk_te))
        
        
        #print(epoch,loss.item(),loss_full[-1],acc.item())
        theta1 = list(net1.parameters())[0].detach().cpu().numpy().copy()
        u1 = list(net2.parameters())[0].detach().cpu().numpy().copy()
        dtheta = (np.mean(np.sum((theta1-theta0)**2,1)**0.5))
        du = (np.mean(np.sum((u1-u0)**2,0)**0.5))/alpha*h_dim
        if print_result:
            print('epoch %d'%epoch,'loss (train,test):%.2e;%.2e'%loss_full[-1][:2],'acc:%.6f'%acc.item())
            print('dtheta:',dtheta)
            print('du:',du)
        relative_change.append((dtheta,du))

        #print(epoch,loss.item(),loss_full[-1],acc.item())
        #print('ntk',loss_ntk_tr.item(),loss_ntk_te,acc_ntk_mean.item())

    return relative_change,acc_full,loss_full
