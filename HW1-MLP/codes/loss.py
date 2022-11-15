from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        loss = ((input-target)**2).sum(-1)
        batch_size = len(input)
        loss = loss.sum(-1)/(2*batch_size)
        return loss
        # TODO END

    def backward(self, input, target):
		# TODO START
        '''Your codes here'''
        batch_size = len(input)
        grad = (input-target)/batch_size
        return grad
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        batch_size = len(input)

        numerator = np.exp(input)
        denominator = numerator.sum(-1)
        
        softmax = ((numerator.T)/denominator).T
        loss = -((target*(np.log(softmax))).sum(-1))
        loss = loss.sum(-1)/batch_size
        return loss
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        batch_size = len(input)
        exp_input = np.exp(input)
        exp_denominator = exp_input.sum(-1)
        soft_max = ((exp_input.T)/exp_denominator).T
        
        grad = target-soft_max
        grad = -grad/batch_size

        return grad
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        '''Your codes here'''
        batch_size = len(input)
        x_t = (input * target).sum(-1)
        tmp = ((input.T)-x_t).T

        tmp = tmp + self.margin
        adjust = target * self.margin
        tmp = tmp - adjust
        tmp[tmp<0] = 0
        
        loss = ((tmp.sum(-1)).sum(-1))/batch_size
        return loss
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        batch_size = len(input)
        x_t = (input * target).sum(-1)
        tmp = ((input.T)-x_t).T
        tmp += self.margin
        adjust = target * self.margin
        tmp = tmp - adjust

        tmp[tmp>0] = 1
        tmp[tmp<1] = 0

        adjust = -tmp.sum(-1)
        adjust = ((target.T)*adjust).T

        tmp = tmp + adjust
        tmp = tmp/batch_size
        return tmp
        # TODO END

