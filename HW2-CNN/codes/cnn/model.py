# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features, epsilon=1e-5, momentum=1e-3):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# Parameters
		self.weight = nn.Parameter(torch.ones(num_features , device = device))
		self.bias = nn.Parameter(torch.zeros(num_features , device = device))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features, device= device))
		self.register_buffer('running_var', torch.zeros(num_features, device= device))
		
		# Initialize your parameter
		self.epsilon = epsilon
		self.momentum = momentum

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training == False:
			_z = (input - self.running_mean[None, :, None, None]) / torch.sqrt(self.running_var[None, :, None, None] + self.epsilon)
			_output = _z * self.weight[None, :,None, None] + self.bias[None, :, None, None]
			return _output
		_mean = input.mean(dim = [0, 2, 3])
		_variance = input.var(dim=[0, 2, 3], unbiased=False)
		self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * _mean
		self.running_var = self.momentum * self.running_var + (1 - self.momentum) * _variance
		
		z = (input - _mean[None, :, None, None]) / torch.sqrt(_variance[None, :, None, None] + self.epsilon)
		output = z * self.weight[None, :, None, None] + self.bias[None, :, None, None]      
		return output
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if self.training == False:
			return input
		num_channel = len(input[0])
		multiplier = torch.bernoulli((1 - self.p) * torch.ones(num_channel, device= device)) * (1 / (1 - self.p)) # Drop out entire channel
		return input * multiplier[None, :, None, None]
	# TODO END

class Dropout1d(nn.Module):
	def __init__(self, p=0.5):
		super().__init__()
		self.p = p

	def forward(self, input):
		if self.training == False:
			return input
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		multiplier = torch.bernoulli((1 - self.p) * torch.ones_like(input, device= device)) *  (1 / (1 - self.p)) 
		return input * multiplier

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		channels_num_1 = 32
		channels_num_2 = 64
		self.net = nn.Sequential(
			nn.Conv2d(3,channels_num_1,kernel_size=5,stride=1,padding=2),
			BatchNorm2d(channels_num_1),
			nn.ReLU(),
			Dropout(drop_rate),
			#Dropout1d(drop_rate),
			nn.MaxPool2d(2,2),
			nn.Conv2d(channels_num_1,channels_num_2,kernel_size=5,stride=1,padding=2),
			BatchNorm2d(channels_num_2),
			nn.ReLU(),
			Dropout(drop_rate),
			#Dropout1d(drop_rate),
			nn.MaxPool2d(2,2)
		)
		self.linear = nn.Linear(8*8*channels_num_2,10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		feature = self.net(x)
		feature_flatten = feature.reshape(feature.shape[0],-1)
		logits = self.linear(feature_flatten)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
