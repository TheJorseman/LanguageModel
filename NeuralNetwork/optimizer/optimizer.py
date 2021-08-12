import numpy as np

class Optimizer(object):
  """
  Clase base que contiene lo necesario para generar un optimizador
  """
  def __init__(self, model, loss, lr=0.005):
    """
    El constructor requiere de: 
      model (Module) : Modelo que se quiere optimizar.
      loss  (Module) : Funcion de perdida a utilizar.
      lr    (float)  : Learning Rate para hacer el Gradiend Descend.
    """
    self.model = model
    self.parameters = model.parameters()
    self.loss = loss
    self.lr = lr
  
  # Método para poner en cero los gradientes (No utilizado)
  def zero_grad(self):
    return
  # Método para realizar el step o el backpropagation + gradiend descend.
  def step(self):
    return