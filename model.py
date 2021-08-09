from sklearn.model_selection import train_test_split
import numpy as np

from tools.clean_corpus import clean_raw_corpus
from tools.bos_eos import insert_bos_eos
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
from NeuralNetwork.functions.softmax import Softmax
# Optimizer 
from NeuralNetwork.optimizer.sgd import SGD

raw_corpus = open("corpus.txt","r", encoding="utf-8")
raw_corpus_list = list(raw_corpus.readlines())

clean_corpus_str = clean_raw_corpus(raw_corpus_list)
new_corpus = insert_bos_eos(clean_corpus_str)

vocabulary = create_vocabulary(new_corpus)
print(vocabulary)
print("Longitud del Vocabulario: ",len(vocabulary))

bigrams = create_bigrams(new_corpus, vocabulary)
print(bigrams[:10])

bigrams_train, bigrams_test = train_test_split(bigrams, test_size=0.2,shuffle=True)

batch_size = 8
learning_rate = 0.1
epochs = 100

dataset = LanguageDataset(vocabulary, bigrams_train, bigrams_test)
train_dataset = dataset.get_train_dataset()

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
n_input = len(vocabulary)
print(n_input)

model = Sequential(
                Linear(n_input,100, bias=False),
                Linear(100,300), 
                Tanh(),
                Linear(300,n_input),
                Softmax()
                )
loss = CrossEntropyLoss()

optimizer = SGD(model, loss, lr=learning_rate, batch_size=batch_size)

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

for epoch in range(epochs):
  train(train_dataloader, model, loss, optimizer, epoch, return_if_zero=False)

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