import torch
from torch import optim, nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random



# Linearization for two-level tanh activated NN
def ntk_v3(model, model0, x):
    h = lambda x:model[0](x)
    h0 = lambda x:model0[0](x)

    out_h0 = h0(x) # b x h
    batch = x.shape[0]
    hidden = out_h0.shape[1]
    single_out_size = out_h0.shape[1]
    d_h0 = 1-out_h0**2 # b x h
    
    delta_theta = model[0][0].weight.data - model0[0][0].weight.data # h x X
    theta_dot_x = torch.torch.matmul(x,delta_theta.t()) # b x h
    
    ker_tangent = d_h0*theta_dot_x # b x h
    u_now =  model[1].weight.data # Y x h
    tangent_term = torch.matmul(ker_tangent,model[1].weight.data.t()) #b x Y
    
    random_term = torch.matmul(out_h0 , (u_now - model0[1].weight.data).t()) # b x Y
    if not (model[1].bias is None):
        bias_term = model[1].bias.data.view(1,-1)
    else:
        bias_term = 0
    init_term = torch.matmul(out_h0 , (model0[1].weight.data.t()))

    return init_term+random_term+tangent_term+bias_term