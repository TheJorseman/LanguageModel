import numpy as np

from NeuralNetwork.module import Module

class Linear(Module):
  def __init__(self, input_size, output_size, norm=1, bias=True):
    super(Linear,self).__init__()
    self.is_bias = bias
    self.w = np.random.rand(input_size, output_size)*1*(1/np.sqrt(input_size))
    bias_function = np.ones if bias else np.zeros
    self.b = bias_function(output_size)

  def forward(self, x):
    return np.tensordot(x,self.w, axes=(-1,0)) + self.b

  def backward(self):
    return self.args[0]

  def update_values(self, dw, db):
    self.w -= dw
    self.b -= db * int(self.is_bias) 