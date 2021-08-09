import numpy as np

from NeuralNetwork.optimizer.vanillagd import VanillaGD

class SGD(VanillaGD):
  def __init__(self, model, loss, lr=0.005, batch_size=1):
    super(SGD,self).__init__(model, loss, lr=lr)
    self.batch_size = batch_size

  def gradient_descend(self, module, d_out):
    out_d_out = np.dot(d_out, module.w.T)
    hl = module.backward()
    for i in range(self.batch_size):
      dw = np.dot(np.expand_dims(hl[i].T, 0).T, np.expand_dims(d_out[i], 0))
      db = d_out.sum(0)
      dw += self.reg*module.w
      module.update_values(self.lr*dw, self.lr*db)
    return out_d_out