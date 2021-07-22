import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# things we need for Tensorflow
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

import processIntents

def trainModel(retrain=False):
  # create our training data
  training = []
  # create an empty array for our output
  output_empty = [0] * len(processIntents.classes)
  # training set, bag of words for each sentence
  for doc in processIntents.documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word - create base word, in attempt to represent related words
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in processIntents.words:
      bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[processIntents.classes.index(doc[1])] = 1

    training.append([bag, output_row])
  # shuffle our features and turn into np.array
  random.shuffle(training)
  training = np.array(training)
  # create train and test lists. X - patterns, Y - intents
  train_x = list(training[:, 0])
  train_y = list(training[:, 1])

  model = Sequential()
  model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(len(train_y[0]), activation='softmax'))

  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

  model.fit(np.array(train_x), np.array(train_y), epochs=1500, batch_size=30, verbose=1)
  model.save('saved_model/my_model')

  return model
