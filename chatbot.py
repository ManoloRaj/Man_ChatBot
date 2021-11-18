import random
import json
import pickle 
import numpy as np

from nltk.stem import WordNetLemmatizer
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

leamtizer = WordNetLemmatizer

intents = json.loads(open('data.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?',' ']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        words_list = nltk.word_tokenize(pattern)
        words.extend(words_list)
        documents.append((words_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

