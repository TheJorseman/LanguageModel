import numpy as np

from NeuralNetwork.module import Module

class CrossEntropyLoss(Module):
  def forward(self, input, target):
    x = -np.log(self.entropy(input, target))
    return np.array([x])
  
  def backward(self, softmax=False):
    return self.derivate(*self.args, softmax=softmax)

  def derivate(self, input, target, softmax=False):
    if softmax:
      out = self.derivate_softmax(input, target)
      return out
    return input - target

  def derivate_softmax(self, input, target):
    return input - target

  def entropy(self, input, target):
    x = np.tensordot(input, target, axes=(-1,-1))
    return np.diag(x)