import numpy as np

from NeuralNetwork.data.dataset import Dataset

class XOR_DataLoader(Dataset):
  """
  Se creo este método para probar la interacción de todos los módulos definidos asi como se vio en clase.
  """
  def __init__(self):
    self.x = np.array([[0,1],[1,0],[0,0],[1,1]])
    self.y = np.array([[0,1],[0,1],[1,0],[1,0]])
    self.data_len = len(self.x)
  def __getitem__(self, i):
    return {'input': np.array([self.x[i]]), 'output': np.array([self.y[i]])}

  def __iter__(self):
    return iter([{'input': np.array([self.x[d]]), 'output': np.array([self.y[d]])} for d in range(len(self.x)) ])

  def __len__(self):
    return self.data_len