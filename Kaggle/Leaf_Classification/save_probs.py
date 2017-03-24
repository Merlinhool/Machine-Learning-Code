import preprocess
import MyPredict

import pickle
import random
import math
import os
import sys
import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

#import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten, Merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adamax
from keras.utils.np_utils import to_categorical
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

SIZE = 128
BATCH_SIZE = 512
learning_rate_decay = math.pow(0.2,0.1)
CLASS_NUMBER = 99

def build_model(learning_rate = 0.001, reg_rate = 0.001, input_size = SIZE):
    cnn = Sequential()
    cnn.add(Convolution2D(8, 5, 5, border_mode='same', activation='relu', input_shape=(1, input_size, input_size)))
    #cnn.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu'))
    #cnn.add(BatchNormalization())
    #cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu'))
    #cnn.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    #cnn.add(BatchNormalization())
    #cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Flatten())
    cnn.add(Dropout(0.5))

    mlp = Sequential()
    mlp.add(Flatten(input_shape=(3, 8, 8)))

    model = Sequential()
    model.add(Merge([cnn, mlp], mode='concat'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(99, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    model = build_model()
    model.load_weights('modelDump/5-0.00536323449841-1.0.h5')
    _ = preprocess.load_data('data/train.csv', 'train')
    te_p, te_f, te_ids = preprocess.load_data('data/test.csv', 'test')
    #with open('ids.pkl','wb') as f:
        #pickle.dump(te_ids, f)
    prob = model.predict_proba([te_p, te_f])
    probs, ids = MyPredict.average_probs(prob, te_ids, 'cnn_pred_n.csv')
    with open('ids.pkl','wb') as f:
        pickle.dump(ids, f)
    with open('probs.pkl','wb') as f:
        pickle.dump(probs, f)
    #MyPredict.output_to_csv(probs, ids, 'for_test.csv')

def output_to_csv(path, pred_path):
    ids = pickle.load(open('ids.pkl','rb'))
    probs = pickle.load(open(path,'rb'))
    MyPredict.output_to_csv(probs, ids, pred_path)

if __name__ == '__main__':
    main()
