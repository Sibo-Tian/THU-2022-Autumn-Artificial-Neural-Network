# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features, epsilon=1e-5, momentum=1e-3):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# Parameters
		self.weight = nn.Parameter(torch.ones(num_features, device = device))
		self.bias = nn.Parameter(torch.zeros(num_features, device = device))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features, device = device))
		self.register_buffer('running_var', torch.zeros(num_features, device = device))
		
		# Initialize your parameter
		self.epsilon = epsilon
		self.momentum = momentum

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training == False:
			_z = (input - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
			_output = _z * self.weight + self.bias
			return _output
		_mean = input.mean(dim=0)
		_variance = input.var(dim=0, unbiased=False)
		self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * _mean
		self.running_var = self.momentum * self.running_var + (1 - self.momentum) * _variance
		
		z = (input - _mean) / torch.sqrt(_variance + self.epsilon)
		output = z * self.weight + self.bias      
		return output
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training == False:
			return input
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		multiplier = torch.bernoulli((1 - self.p) * torch.ones_like(input, device = device)) * (1 / (1 - self.p))
		return input * multiplier
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		hidden_dim = 1024
		self.net = nn.Sequential(nn.Linear(3072, hidden_dim),
							BatchNorm1d(hidden_dim),
							nn.ReLU(),
							Dropout(drop_rate),
							nn.Linear(hidden_dim, 10)
							)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = self.net(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
