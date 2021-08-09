class Dataset():
  def __init__(self, *args):
    self.args = args
  
  def __getitem__(self, i):
    return []

  def __len__(self):
    return 0
    