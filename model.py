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

raw_corpus = open("reduced_corpus.txt","r", encoding="utf-8")
raw_corpus_list = list(raw_corpus.readlines())

clean_corpus_str = clean_raw_corpus(raw_corpus_list)
new_corpus = insert_bos_eos(clean_corpus_str)
new_corpus = get_reduced_corpus(new_corpus, 1000)
vocabulary = create_vocabulary(new_corpus)
print(vocabulary)
print("Longitud del Vocabulario: ",len(vocabulary))

bigrams = create_bigrams(new_corpus, vocabulary)
print(bigrams[:10])

bigrams_train, bigrams_test = train_test_split(bigrams, test_size=0.3,shuffle=True)

batch_size = 4
learning_rate = 0.01
epochs = 500

dataset = LanguageDataset(vocabulary, bigrams_train, bigrams_test)
train_dataset = dataset.get_train_dataset()

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
n_input = len(vocabulary)
print(n_input)

model = Sequential(
                Linear(n_input,100, bias=False),
                Linear(100,300), 
                ReLU(),
                BatchNorm1d(),
                Linear(300,n_input),
                Softmax()
                )

model_name = "model-final-1.bin"
if model_name in os.listdir("."):
  print("Modelo cargado")
  model = pickle.load(open(model_name,"rb"))

loss = CrossEntropyLoss()

optimizer = SGD(model, loss, lr=learning_rate, batch_size=batch_size)
#optimizer = VanillaGD(model, loss, lr=learning_rate)

def calculate_metric(output, target):
  return ((np.round(output)-target)**2).sum(0)

def train(dataset, model, loss, optimizer, epoch, return_if_zero=False):
  for i in range(len(dataset)):
    data = dataset[i]
    input = data["input"]
    target = data["output"]
    output = model(input)
    loss_value = loss(output, target)
    entropy = loss.entropy(output, target)
    error = calculate_metric(output, target)
    accuracy = 100-error
    print("Epoch ", epoch)
    print("Accuracy", accuracy)
    print("Loss = {}, Entropy = {} , Error = {}".format(loss_value.mean(), entropy.mean(), error.sum()))
    optimizer.step()
    print(sum(error))
    print("Procesado {} de {}".format(i, len(dataset)))

for epoch in range(epochs):
  train(train_dataloader, model, loss, optimizer, epoch, return_if_zero=False)
  if epoch % 50 == 0:
    model_name = "model-epoch-{}.bin".format(epoch)
    model_dump = open(model_name, "wb")
    pickle.dump(model, model_dump)

def test(dataset, model, loss):
  for i in range(len(dataset)):
    data = dataset[i]
    input = data["input"]
    target = data["output"]
    output = model(input)
    loss_value = loss(output, target)
    entropy = loss.entropy(output, target)
    error = calculate_metric(output, target)
    accuracy = 100-error
    print("Accuracy", accuracy)
    print("Loss = {}, Entropy = {} , Error = {}".format(loss_value, entropy, error))
  print("Total accuracy", accuracy)

dataset_test = dataset.get_test_dataset()
test(dataset_test, model, loss)

model_dump = open("model-final.bin", "wb")
pickle.dump(model, model_dump)
import pdb;pdb.set_trace()