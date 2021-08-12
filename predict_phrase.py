from sklearn.model_selection import train_test_split
import numpy as np

from tools.clean_corpus import clean_raw_corpus
from tools.bos_eos import insert_bos_eos, get_reduced_corpus, out_of_vocabulary_token
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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("phrase", help="Frase que se quiere calcular la probabilidad",
                    type=str)
args = parser.parse_args()

loss = CrossEntropyLoss()

def calc_probability(vocabulary, one_hot_m, model, words):
  corpus = clean_raw_corpus(["<BOS> "+words+" <EOS>"])
  default = vocabulary.get(out_of_vocabulary_token)
  probability = 1.0
  words = corpus.split(" ")
  for i in range(len(words)-1):
    n_word = vocabulary.get(words[i+1], default)
    vec = one_hot_m[vocabulary.get(words[i], default)]
    output = model(vec).squeeze()
    probability *= output[n_word]
  return probability


vocabulary = pickle.load(open("vocabulary", "rb"))
model = pickle.load(open("model-final.bin", "rb"))

one_hot = np.identity(len(vocabulary))

probability = calc_probability(vocabulary, one_hot, model, args.phrase)
print(probability)