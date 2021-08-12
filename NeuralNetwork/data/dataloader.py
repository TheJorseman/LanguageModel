import numpy as np

class DataLoader(object):
  """
  Clase para cargar los datos en batches y hacer mas eficiente el procesamiento de los datos.
  """
  def __init__(self, dataset, batch_size=1):
    self.dataset = dataset
    self.batch_size = batch_size
  
  def __getitem__(self, i):
    index = i * self.batch_size
    input = []
    output = []
    for i in range(index, index + self.batch_size):
      data = self.dataset[i]
      input.append(np.squeeze(data['input']))
      output.append(np.squeeze(data['output']))
    return {'input': np.array(input), 'output': np.array(output)}

  def __len__(self):
    return len(self.dataset)//self.batch_size