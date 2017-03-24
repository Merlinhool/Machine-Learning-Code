import pickle as pkl
import random
import math
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

#import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

SIZE = 28
BATCH_SIZE = 256
train_val_rate = 5000.0 / 42000.0
learning_rate_decay = math.pow(0.1, 1.0/20.0)


def load_data(path, flag):
    print('Loading ' + flag + ' data...')
    data = pd.read_csv(path).values
    if flag == 'train':
        y = data[:,0]
        #y = to_categorical(data[:,0], nb_classes = 10)
        tmp = (data[:,1:].reshape((data.shape[0], 1, SIZE, SIZE)) / 255.0).astype(np.float32)
        x = tmp - tmp.mean(axis=0) 
        xt,xv,yt,yv_classes = train_test_split(x, y, test_size = train_val_rate, random_state=0)
        yt = to_categorical(yt, nb_classes = 10)
        yv = to_categorical(yv_classes, nb_classes = 10)
        return xt,xv,yt,yv, yv_classes
    else:
        x = (data.reshape((data.shape[0], 1, SIZE, SIZE)) / 255.0).astype(np.float32)
        x = x - x.mean(axis=0)
        return x

# learning rate decay
def scheduler(epoch):
    print(REG_RATE)
    return math.pow(learning_rate_decay, epoch) * LEARNING_RATE

def build_model(learning_rate = 0.001, reg_rate = 0.001, input_size = SIZE, complie=True):
    model = Sequential()

    model.add(Convolution2D(128, 3, 3, init='he_normal', activation='relu', border_mode='same', W_regularizer=l2(reg_rate), input_shape=(1, input_size, input_size)))
    model.add(Convolution2D(128, 3, 3, init='he_normal', activation='relu', border_mode='same', W_regularizer=l2(reg_rate)))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, init='he_normal', activation='relu', border_mode='same', W_regularizer=l2(reg_rate)))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

    model.add(Convolution2D(256, 3, 3, init='he_normal', activation='relu', border_mode='same', W_regularizer=l2(reg_rate)))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, init='he_normal', activation='relu', border_mode='same', W_regularizer=l2(reg_rate)))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, init='he_normal', activation='relu', border_mode='same', W_regularizer=l2(reg_rate)))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(256, init = 'he_normal', activation='relu', W_regularizer=l2(reg_rate)))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, init = 'he_normal', activation='relu', W_regularizer=l2(reg_rate)))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, init = 'he_normal', activation=None, W_regularizer=l2(reg_rate)))
    model.add(Activation('softmax'))

    if not complie:
        return model

    #opt = Adam(lr = learning_rate, callbacks=[lr_rate])
    opt = Adam(lr = learning_rate)
    loss='categorical_crossentropy'
    met=['accuracy']
    model.compile(optimizer=opt, loss=loss, metrics=met)
    return model

# accuracy: 0.99471
def train_model_without_DA(path, iteration_time, epochs, reg_intv, lr_rate):
    global LEARNING_RATE
    global REG_RATE
    global xt,xv,yt,yv,yv_classes

    if os.path.exists(path):
        #model = build_model(complie = False)
        #model.load_weights(path)
        #return model
        return
    else:
        best = (None, -1)
        for it in range(iteration_time):
            print('No.' + str(it) + '.............................................................')
            #learning_rate = 10**random.uniform(lr[0], lr[1])
            learning_rate = lr_rate
            reg_rate = 10**random.uniform(reg_intv[0], reg_intv[1])
            #reg_rate = 0.001

            REG_RATE = reg_rate
            LEARNING_RATE = learning_rate

            model = build_model(learning_rate, reg_rate)
            print('learning_rate:' + str(learning_rate) + '   reg_rate: ' + str(reg_rate))

            print('Fitting...')
            lr_rate_sch = LearningRateScheduler(scheduler)
            stop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, mode='max')
            save_best = ModelCheckpoint(filepath='modelDump/%d-{val_acc:.5f}.h5' % it, monitor='val_acc', verbose=2, save_best_only=True, save_weights_only=False)
            history = model.fit(xt, yt, batch_size=BATCH_SIZE, nb_epoch=epochs, verbose = 1, validation_data=(xv,yv), callbacks=[lr_rate_sch, stop, save_best])

            #model.save('small-vgg-' + str(i) + '.h5')
            #temp = history.history['val_acc'][-1]
            #print('learning_rate:' + str(learning_rate) + '   reg_rate: ' + str(reg_rate) + '   acc:' + str(temp))
            #if temp > best[1]:
                #best = (model, temp, it, learning_rate, reg_rate, history)
            #backup.append((history, learning_rate, reg_rate))
        open(path, 'w')

        return 
        #print('best model:')
        #print('learning_rate:' + str(best[3]) + '   reg_rate: ' + str(best[4]) + '   val_acc:' + str(best[1]))
        #best[0].save_weights(path)
        #return best[0]

# accuracy: 0.992
def train_model_final(input_model, path, iteration_time, epochs, reg_intv, lr_rate):
    global LEARNING_RATE
    global REG_RATE
    global xt,xv,yt,yv,yv_classes

    if not os.path.exists(path):
        for it in range(iteration_time):
            print('No.' + str(it) + '.............................................................')
            #learning_rate = 10**random.uniform(lr[0], lr[1])
            learning_rate = lr_rate
            reg_rate = 10**random.uniform(reg_intv[0], reg_intv[1])
            #reg_rate = 0.001

            LEARNING_RATE=learning_rate
            REG_RATE=reg_rate

            model = build_model(learning_rate, reg_rate, complie=False)
            model.set_weights(input_model.get_weights())
            datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.15, height_shift_range=0.15, shear_range=0.2, zoom_range=0.3)

            opt = Adam(lr = learning_rate)
            loss='categorical_crossentropy'
            met=['accuracy']
            model.compile(optimizer=opt, loss=loss, metrics=met)

            save_best = ModelCheckpoint(filepath='modelDump/%d-{val_acc:.5f}.h5' % it, monitor='val_acc', verbose=2, save_best_only=True, save_weights_only=True)
            lr_rate_sch = LearningRateScheduler(scheduler)
            stop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=7, mode='max')
            model.fit_generator(datagen.flow(xt, yt, batch_size=BATCH_SIZE,seed=it),
                                          samples_per_epoch=xt.shape[0],
                                          nb_epoch=epochs,
                                          verbose=1,
                                          initial_epoch=32,
                                          callbacks=[save_best, lr_rate_sch, stop],
                                          validation_data=(xv, yv))
        # a sign that final models are trained
        open(path, 'w')

# train about 10 models and select one with highest val_acc, or select all the models of which the val_acc > acc_threshold
def filter_models(path, one = True, acc_threshold = 0.995, save_weights_only=False):
    if os.path.exists(path):
        print('Loading filtered model...')
        return load_model(path)
    else:
        print('Filtering model...')
        os.chdir('modelDump')
        names = os.listdir()
        best = [(None, -math.inf)] * 20
        for name in names:
            if name[1] == '-':
                it = int(name[0]) - 1
                acc = float(name[2:9])
            else:
                it = int(name[0:2]) - 1
                acc = float(name[3:10])
            if acc > best[it][1]:
                best[it] = (name, acc)

        print(best)

        best_model = (None, -math.inf)
        models = []
        for (name,acc) in best:
            print(name, acc)
            if one:
                if acc > best_model[1]:
                    if save_weights_only:
                        t= build_model()
                        t.load_weights(name, by_name=False)
                    else:
                        t = load_model(name)
                    best_model = (t, acc)
            else:
                if acc > acc_threshold:
                    models.append(load_model(name))

        if one:
            os.chdir('..')
            best_model[0].save(path)
            return best_model[0]
        else:
            return models

def main():
    global LEARNING_RATE
    global REG_RATE
    global xt,xv,yt,yv,yv_classes
    xt,xv,yt,yv,yv_classes = load_data('data/train.csv', 'train')
    test = load_data('data/test.csv', 'test')

    # accuracy: 0.99471 (with Data Augmentation, kaggle public leaderboard)
    lr = 0.002
    reg = (math.log(0.0003,10),-6)
    model_path = 'model_withoutDA_trained.h5'
    train_model_without_DA(model_path, 20, 32, reg, lr_rate=lr)
    filter_model_path = 'model_withoutDA_filtered.h5'
    model = filter_models(filter_model_path)

    #print('Predicting...')
    #pred = model.predict_classes(test, batch_size=BATCH_SIZE)
    #result = pd.DataFrame([[index+1, x] for index, x in enumerate(pred)])
    #result.to_csv('final_pred_withoutDA.csv', header = ['ImageId','Label'], index=False)

    print('Data Augmentation version...')
    # accuracy: 0.992 (with Data Augmentation, kaggle public leaderboard)

    final_path = 'final_models_trained.h5'
    models = train_model_final(model, final_path, 10, 64+32, reg, lr_rate=lr)
    filter_model_path = 'model_withDA_filtered.h5'
    model = filter_models(filter_model_path, save_weights_only = True)

    print('Predicting...')
    pred = model.predict_classes(test, batch_size=BATCH_SIZE)
    result = pd.DataFrame([[index+1, x] for index, x in enumerate(pred)])
    result.to_csv('final_pred_withDA.csv', header = ['ImageId','Label'], index=False)

if __name__ == '__main__':
    main()
