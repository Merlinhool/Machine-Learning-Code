import preprocess
import MyPredict

import pickle as pkl
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
from keras.optimizers import Adamax, Adam, RMSprop
from keras.utils.np_utils import to_categorical
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

SIZE = 128
BATCH_SIZE = 256
learning_rate_decay = math.pow(0.2,0.1)
LEARNING_RATE_DECAY = 0.85
CLASS_NUMBER = 99

def scheduler(epoch):
    print(REG_RATE)
    return math.pow(learning_rate_decay, epoch) * LEARNING_RATE

def build_model(learning_rate = 0.001, reg_rate = 0.001, input_size = SIZE):
    cnn = Sequential()
    cnn.add(Convolution2D(8, 5, 5, border_mode='same', init='he_normal', activation='relu', input_shape=(1, input_size, input_size)))
    #cnn.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu'))
    #cnn.add(BatchNormalization())
    #cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Convolution2D(32, 5, 5, border_mode='same', init='he_normal', activation='relu'))
    #cnn.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    #cnn.add(BatchNormalization())
    #cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Flatten())
    cnn.add(Dropout(0.5))

    another = Sequential()
    another.add(Flatten(input_shape=(3, 8, 8)))

    model = Sequential()
    model.add(Merge([cnn, another], mode='concat'))
    model.add(Dense(128, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(99, init='he_normal', activation='softmax'))

    #optim = Adam(lr=learning_rate)
    optim = RMSprop(lr=learning_rate)

    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def MyModelFit(model, save_path, model_id, train_gen, learning_rate, samples_per_epoch, nb_epoch, verbose, validation_data, stop_epochs):

    #model.optimizer.lr.set_value(learning_rate)
    print('model learning rate:' + str(model.optimizer.lr.get_value()))
    
    val_data = validation_data[0]
    val_label = validation_data[1]
    
    acc, val_acc, loss, val_loss = [], [], [], []
    history = dict({'acc': acc, 'val_acc':val_acc, 'loss':loss, 'val_loss':val_loss})
    min_val_loss = math.inf
    to_val_acc = None
    last_min = -1
    best_weights = None
    flag_first_inc_lr = False
    for it in range(nb_epoch):
        print('Iteration ' + str(model_id) + ' Epoch ' + str(it) + '/' + str(nb_epoch) + '..........')
        sys.stdout.flush()
        batch_nums = 0
        acc_arr = []
        loss_arr = []
        for (pics, y1), (fea, y2) in train_gen:
            #print(str(batch_nums) + '/' + str(samples_per_epoch) + ':')
            #sys.stdout.flush()
            #assert(np.all(y1==y2))
            met = model.train_on_batch([pics, fea], y1)
            acc_arr.append(met[1])
            loss_arr.append(met[0])
            batch_nums += 1
            if batch_nums >= samples_per_epoch:
                break

        met = model.evaluate(val_data, val_label, verbose=0)
        acc = np.array(acc_arr).mean()
        loss = np.array(loss_arr).mean()
        history['acc'].append(acc)
        history['loss'].append(loss)
        history['val_loss'].append(met[0])
        history['val_acc'].append(met[1])

        print('acc : ' + str(acc) + ' loss:' + str(loss) + ' val_acc:' + str(met[1]) + ' val_loss:' + str(met[0]))
        #sys.stdout.flush()

        if (not flag_first_inc_lr) and met[1] > 0.1:
            model.optimizer.lr.set_value(0.005)
            flag_first_inc_lr = True

        if it >= 30 and met[1] < 0.5:
            break
        if it >= 50 and met[1] < 0.9:
            break

        if met[1] > 0.9 and met[0] < min_val_loss:
            print('...imporve train_loss from ' + str(min_val_loss) + ' to ' + str(met[0]) + ' . (val_acc: ' + str(met[1]) + ')')
            #model.save_weights(save_path)
            best_weights = model.get_weights()
            min_val_loss, to_val_acc = met
            last_min = it
        elif it > 30:
            if it - 10 > last_min:
                #model.optimizer.lr.set_value(model.optimizer.lr.get_value() * LEARNING_RATE_DECAY)
                print('model learning rate:' + str(model.optimizer.lr.get_value()))
                model.optimizer.lr.set_value(np.array(model.optimizer.lr.get_value() * LEARNING_RATE_DECAY, dtype='float32'))
                print('model learning rate:' + str(model.optimizer.lr.get_value()))

            if it - stop_epochs > last_min:
                break

        if it % 50 == 0 and best_weights != None:
            new_model = build_model(learning_rate, 0)
            new_model.set_weights(best_weights)
            new_model.save('modelDump/' + str(model_id) + '-' + str(min_val_loss) + '-' + str(to_val_acc) + '-iteration' + str(it) + '.h5')


    if to_val_acc != None:
        #model.load_weights(save_path)
        model.set_weights(best_weights)
        model.save('modelDump/' + str(model_id) + '-' + str(min_val_loss) + '-' + str(to_val_acc) + '.h5')
    
    return history
        

def train_model(path, iteration_time, epochs, reg_intv, lr_rate):
    global LEARNING_RATE
    global REG_RATE
    global tr_p, tr_f, tr_y, va_p, va_f, va_y, label_encoder, tr_label, va_label, useful_iter

    if os.path.exists(path):
        return
    else:
        useful_iter = set()
        best = (None, -1)
        for it in range(iteration_time):
            print('No.' + str(it) + '.............................................................')
            learning_rate = lr_rate
            reg_rate = 10**random.uniform(reg_intv[0], reg_intv[1])
            reg_rate = 0
            learning_rate = 0.0003

            REG_RATE = reg_rate
            LEARNING_RATE = learning_rate
            SAMPLES_PER_EPOCH = (len(tr_p) // BATCH_SIZE) + 1

            print('Build model...')
            model = build_model(learning_rate, reg_rate)

            print('Preprocess data...')
            image_gen_args = dict(rotation_range=90.,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  zoom_range=0.2)
            image_gen = ImageDataGenerator(**image_gen_args)
            fea_gen = ImageDataGenerator()

            seed = 0
            image_gen.fit(tr_p, augment=True, seed=seed)
            fea_gen.fit(tr_f, augment=False, seed=seed)

            train_gen = zip(image_gen.flow(tr_p,tr_label, batch_size=BATCH_SIZE, seed=seed), fea_gen.flow(tr_f,tr_label,batch_size=BATCH_SIZE, seed=seed))
            
            print('model compile...')
            opt = Adamax(lr = learning_rate)
            loss='categorical_crossentropy'
            met=['accuracy']
            model.compile(optimizer=opt, loss=loss, metrics=met)
            
            lr_rate_sch = LearningRateScheduler(scheduler)
            stop = EarlyStopping(monitor='val_loss', patience=15, mode='min')
            save_best = ModelCheckpoint(filepath='modelDump/%d-{val_loss:.9f}-{val_acc:.5f}-%f.h5' % (it,reg_rate), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True)

            print('Fitting...')
            print('learning_rate:' + str(learning_rate) + '   reg_rate: ' + str(reg_rate))
            sys.stdout.flush()
            save_path = 'cnn_weights_tmp.h5'
            history = MyModelFit(model, save_path, it, train_gen, learning_rate = learning_rate, samples_per_epoch = SAMPLES_PER_EPOCH, nb_epoch=epochs, verbose = 1, validation_data=([va_p, va_f], va_label), stop_epochs = 30)
            #history = model.fit_generator(train_gen, samples_per_epoch = SAMPLES_PER_EPOCH, nb_epoch=epochs, verbose = 1, validation_data=([va_p, va_f], va_label), callbacks=[lr_rate_sch, stop, save_best])
            #history = history.history
            mets = ['acc', 'loss', 'val_acc', 'val_loss']
            for name in mets:
                history[name] = np.array(history[name])
                print(name + ':    ' + str(history[name].min()) + '   ' + str(history[name].max()))

            if history['val_acc'].max() > 0.8:
                useful_iter.add(it)

        open(path, 'w')

        return

def filter_models(n, path, loss_threshold = 0.095):
    if os.path.exists(path):
        print('Loading filtered model...')
        t = build_model()
        t.load_weights(path)
        return t
        #return build_model().load_weights(path)
    else:
        global useful_iter
        print('Filtering model...')
        os.chdir('modelDump')
        names = os.listdir()
        best = []
        for i in range(n):
            best.append((None, math.inf))
        for name in names:
            lis = name.split('-')
            it = int(lis[0]) - 1
            loss = float(lis[1])
            #val_acc = float(lis[2])
            if (it in useful_iter) and loss < best[it][1]:
                best[it] = (name, loss)

        best_model = (None, math.inf)
        models = []
        for (name,loss) in best:
            if loss < best_model[1]:
                print('improve loss to %f' % loss)
                best_model = (name, loss)

        t = build_model()
        t.load_weights(best_model[0], by_name=False)
        os.chdir('..')
        t.save_weights(path)
        return t

def main():
    global LEARNING_RATE
    global REG_RATE
    global tr_p, tr_f, tr_y, va_p, va_f, va_y, label_encoder, tr_label, va_label
    tr_p, tr_f, tr_y, va_p, va_f, va_y, label_encoder = preprocess.load_data('data/train.csv', 'train')

    tr_label = to_categorical(tr_y, CLASS_NUMBER)
    va_label = to_categorical(va_y, CLASS_NUMBER)
    #test = load_data('data/test.csv', 'test')

    lr = 0.001
    reg = (-4,-8)
    iteration_time = 10
    model_path = 'model_withoutDA_trained.h5'
    train_model(model_path, iteration_time, 300, reg, lr_rate=lr)

    # choose the best model from the 10 trained models
    filter_model_path = 'model_withoutDA_filtered.h5'
    model = filter_models(iteration_time, filter_model_path)

    te_p, te_f, te_ids = preprocess.load_data('data/test.csv', 'test')
    prob = model.predict_proba([te_p, te_f])
    MyPredict.predict(prob, te_ids, 'cnn_pred.csv')
    return model, te_p, te_f

if __name__ == '__main__':
    main()
