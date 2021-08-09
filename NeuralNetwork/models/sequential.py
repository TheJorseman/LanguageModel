from NeuralNetwork.module import Module

class Sequential(Module):
  def __init__(self, *args):
    super(Sequential,self).__init__()
    self.module_list = list(args)

  def forward(self, x):
    output = x
    for module in self.module_list:
      output = module(output)
    return output