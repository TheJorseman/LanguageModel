from sklearn.model_selection import train_test_split
import numpy as np

from tools.clean_corpus import clean_raw_corpus
from tools.bos_eos import insert_bos_eos, get_reduced_corpus
from tools.vocabulary import create_vocabulary
from tools.bigrams import create_bigrams
# Data
from NeuralNetwork.data.language import LanguageDataset
from NeuralNetwork.data.dataloader import DataLoader
# Modelos
from NeuralNetwork.models.linear import Linear
from NeuralNetwork.models.sequential import Sequential
# Loss
from NeuralNetwork.loss.crossentropy import CrossEntropyLoss
# Act F
from NeuralNetwork.functions.tanh import Tanh
from NeuralNetwork.functions.relu import ReLU
from NeuralNetwork.models.batch_normalization import BatchNorm1d
from NeuralNetwork.functions.softmax import Softmax
# Optimizer 
from NeuralNetwork.optimizer.sgd import SGD
from NeuralNetwork.optimizer.vanillagd import VanillaGD
import pickle
import os

loss = CrossEntropyLoss()

def calculate_metric(output, target):
  return ((np.round(output)-target)**2).sum(0)

def test(dataset, model, loss):
  for i in range(len(dataset)):
    data = dataset[i]
    input = data["input"]
    target = data["output"]
    output = model(input)
    loss_value = loss(output, target)
    entropy = loss.entropy(output, target)
    error = calculate_metric(output, target)
    #accuracy = 100-error
    #print("Accuracy", accuracy)
    print("Loss = {}, Entropy = {} , Error = {}".format(loss_value.mean(), entropy.mean(), error.sum()))
  #print("Total accuracy", accuracy)


dataset_test = pickle.load(open("test_dataset", "rb"))
vocabulary = pickle.load(open("vocabulary", "rb"))
model = pickle.load(open("model-final.bin", "rb"))

test(dataset_test, model, loss)
