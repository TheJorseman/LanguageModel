import numpy as np

np.random.seed(0)

class Module(object):
  def __init__(self):
    self.module_list = []
    self.args = None
    self.apply_backward = True

  def forward(self):
    return

  def backward(self):
    return

  def __call__(self, *args):
    self.args = args
    self.output = self.forward(*args)
    return self.output

  def parameters(self):
    return list(filter(lambda m: m.apply_backward == True, self.module_list))