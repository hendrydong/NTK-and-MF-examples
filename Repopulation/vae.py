import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable

import random

class Model(nn.Module):
	"""docstring for Model"""

	def __init__(self, X_dim ,y_dim,h_dim, Z_dim, batch_size, lr ):
		super(Model, self).__init__()
		self.batch_size = batch_size
		self.X_dim = X_dim
		self.y_dim = y_dim
		self.Z_dim = Z_dim
		self.x2h =  nn.Sequential(nn.Linear(self.X_dim+y_dim, h_dim, bias=True),nn.BatchNorm1d(h_dim),nn.ReLU())
		self.h2mu =nn.Sequential(nn.Linear(h_dim, Z_dim, bias=True),nn.BatchNorm1d(Z_dim),nn.ReLU(),nn.Linear(Z_dim, Z_dim, bias=True))
		self.h2var =nn.Sequential(nn.Linear(h_dim, Z_dim, bias=True),nn.BatchNorm1d(Z_dim),nn.ReLU(),nn.Linear(Z_dim, Z_dim, bias=True))
		self.z2h =nn.Sequential(nn.Linear(Z_dim+y_dim, Z_dim, bias=True),nn.BatchNorm1d(Z_dim),nn.ReLU(),nn.Linear(Z_dim, h_dim, bias=True),nn.BatchNorm1d(h_dim),nn.ReLU())
		self.h2x =  nn.Sequential(nn.Linear(h_dim, self.X_dim,  bias=True),nn.Tanh())
		
		self.solver = optim.Adam(self.parameters(), lr=lr)

	def Q(self,h):
		x = h
		z_mu = self.h2mu(x)
		z_var = self.h2var(x)
		return z_mu, z_var

	def P(self,z):
		h = self.z2h(z)
		return h


	def sample_z(self,mu, log_var):
		eps = Variable(torch.randn(self.batch_size, self.Z_dim)).to(mu.device)
		return mu + torch.exp(log_var / 2) * eps


	def forward(self,X,y):


		h = self.x2h(torch.cat([X,y],1))
		z_mu, z_var = self.Q(h)
		z = self.sample_z(z_mu, z_var)
		self.z = z

		h_hat = self.P(torch.cat([z,y],1))
		x = self.h2x(h_hat)

		return x,z,z_mu, z_var

	def optim(self, X, X_sample, z_mu, z_var):
		self.solver.zero_grad()

		#recon_loss = nn.functional.binary_cross_entropy(X_sample, X, size_average=False) / self.batch_size
		recon_loss = torch.sum((X_sample- X)**2) / self.batch_size
		kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
		loss = recon_loss + kl_loss 
		self.loss = loss
		loss.backward(retain_graph=True)
		self.solver.step()
	def predict(self,y):
		z = Variable(torch.randn(y.shape[0], self.Z_dim)).to(y.device)
		h_hat = self.P(torch.cat([z,y],1))
		x = self.h2x(h_hat)
		return x	
