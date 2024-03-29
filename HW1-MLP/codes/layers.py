import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        super(Relu, self)._saved_for_backward(input)
        return (abs(input)+input)/2
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        input = self._saved_tensor
        input[input <= 0] = 0
        input[input > 1] = 1

        grad_input = input * grad_output
        return grad_input
        # TODO END

class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        super(Sigmoid, self)._saved_for_backward(input)
        res = 1/(1+np.exp(-input))
        return res
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        res = 1/(1+np.exp(-self._saved_tensor))
        grad_input = grad_output*(res*(1-res))
        return grad_input
        # TODO END

class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        super(Gelu, self)._saved_for_backward(input)
        res = 0.5*input*(1+np.tanh(np.sqrt(2/np.pi)*(input+0.044715*np.power(input,3))))
        return res
        # TODO END
    
    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        delta = 1e-5
        input = self._saved_tensor
        res = 0.5*input*(1+np.tanh(np.sqrt(2/np.pi)*(input+0.044715*np.power(input,3))))
        input += delta
        _res = 0.5*input*(1+np.tanh(np.sqrt(2/np.pi)*(input+0.044715*np.power(input,3))))
        grad = (_res-res)/delta
        grad_input = grad_output*grad
        return grad_input
        # TODO END

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        output = np.matmul(input, self.W) + self.b
        super(Linear, self)._saved_for_backward(input)
        return output
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        #backpropagate the gradient
        grad_input = np.matmul(grad_output, self.W.T)

        #update gradient of w and b
        input = self._saved_tensor
        self.grad_W = np.matmul(input.T, grad_output)
        self.grad_b = grad_output.sum(0)
        
        return grad_input
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
