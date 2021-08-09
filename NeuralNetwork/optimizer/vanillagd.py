import numpy as np

from NeuralNetwork.optimizer.optimizer import Optimizer

class VanillaGD(Optimizer):
  def __init__(self, model, loss, lr=0.005, reg=0.0):
    super(VanillaGD,self).__init__(model, loss, lr=lr)
    self.reg = reg

  def step(self):
    modules = list(reversed(self.parameters))
    name = lambda m : m.__class__.__name__
    softmax = False
    if name(modules[0]) == 'Softmax':
      modules = modules[1:]
      softmax = True
    d_out = self.loss.backward(softmax=softmax)
    for module in modules:
      if name(module) == "Linear":
        d_out = self.gradient_descend(module, d_out)
      else:
        d_out = module.backward() * d_out
    return True

  def gradient_descend(self, module, d_out):
    out_d_out = np.dot(d_out, module.w.T)
    hl = module.backward()
    dw = np.dot(hl.T,d_out)
    db = d_out.sum(0)
    dw += self.reg*module.w
    module.update_values(self.lr*dw, self.lr*db)
    return out_d_out
