import numpy as np

from NeuralNetwork.optimizer.optimizer import Optimizer

class VanillaGD(Optimizer):
  """
  Se define el Vanilla Gradiend Descend o Batch Gradiend Descend.
  """
  def __init__(self, model, loss, lr=0.005, reg=0.0):
    super(VanillaGD,self).__init__(model, loss, lr=lr)
    self.reg = reg

  def step(self):
    """
    Se define el conjunto de backpropagation + GD.
    Primero revisa entre todos los parámetros de la red y los pone en manera inversa (para backpropagation)
    Revisa si la ultima capa es Softmax, esto significa que es mas facil calcular la derivada si se usa CrossEntropy.
    Se itera sobre todos los módulos (Capas Lineales y funciones de activación)
    Si hay parámetros entrenables (Capas Lineales Fully Conected) entonces se aplica el gradiend descend.
    Si no, entonces se sigue propagando la derivada con d_out y llamando al backward del modulo (derivada).
    """
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
    """
    Aplica el gradiente descendiente.
    Args:
      module (Module)     : Módulo al que se le va a aplicar el GD.
      d_out  (np.ndarray) : derivada de las capas anteriores.
    """
    # Se obtiene la derivada para las capas próximas.
    out_d_out = np.dot(d_out, module.w.T)
    hl = module.backward()
    # Se obtiene la derivada con respecto a los pesos
    dw = np.dot(hl.T,d_out)
    # Se obtiene la derivada con respecto al bias.
    db = d_out.sum(0)
    # Se aplica la regularización
    dw += self.reg*module.w
    # Se actualizan los valores.
    module.update_values(self.lr*dw, self.lr*db)
    return out_d_out