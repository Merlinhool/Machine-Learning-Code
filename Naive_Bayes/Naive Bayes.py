import math
import numpy as np
import pandas as pd
import random
import pickle
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.externals.joblib as jl
from sklearn import metrics

def getData(path):
    with open(path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        res = u.load()
    return res

def calcPR(pred, res, class_name):
    my = np.array(pred)
    ans = np.array(res)
    psum = 0
    rsum = 0
    for c in class_name:
        i1 = set(np.where(my == c)[0])
        i2 = set(np.where(ans == c)[0])
        cor = len(i1 & i2)
        psum += (cor + 0.0) / len(i1)
        rsum += (cor + 0.0) / len(i2)

    return psum / len(class_name), rsum / len(class_name)

class NaiveBayesClassifier:
    #class_name should be type: [int]
    #type(prior): dict
    #class_name is int
    #epsilon = 1e-9 * np.var(X, axis=0).max()
    def __init__(self, class_name = {0,1}, prior = None, adjustPrior = True):
        self.n_class = len(class_name)
        self.class_name = class_name
        if prior == None:
            self.prior = dict()
            for i in class_name:
                self.prior[i] = 1 / self.n_class
        else: self.prior = prior

    #x is sparse matrix
    #y is list
    def fit(self, x, y):
        xx = x.tocsr()
        #print(xx.todense())
        yy = np.array(y)
        #print(yy)
        self.all_features = dict()
        t_all_features = dict()
        epsilon = 0.0
        for i in self.class_name:
            idx = np.where(yy == i)[0]
            data = xx[idx,:].todense()
            #print(i)
            #print(idx)
            #print(data)
            #print(data.shape)
            #epsilon = 1e-9 * np.var(data, axis=0).max()
            mean = np.array(data.mean(axis=0))
            var = data.var(axis=0)
            epsilon = max(epsilon, np.max(var))
            t_all_features[i] = (self.prior[i], mean, var)
            #print(mean)
            #print(var)
            #print(self.prior[i])
            #print('--------------')

        #print('calculate var...')
        epsilon *= (1e-9)
        for (i,(p,m,v)) in t_all_features.items():
            v[:,:] += epsilon
            #print(v)
            self.all_features[i] = (p,m,np.array(v))

    #X is sparse matrix
    def predict(self, X):
        X = np.array(X.todense())
        log_prob = []
        for (c,(pr,mean,var)) in self.all_features.items():
            print('dealing with class ' + str(c))
            #print('predict : ' + str(c))
            #print(pr)
            #print(mean)
            #print(var)
            prior = np.log(pr)
            p = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            #print(p)
            #print(X)
            #print(X.shape)
            #print(mean)
            #print(mean.shape)
            #print(((X - mean) ** 2))
            #print(((X - mean) ** 2).shape)
            p -= 0.5 * np.sum(((X - mean) ** 2) / (var), axis=1)
            #print(p)
            log_prob.append(prior + p)
        log_prob = np.matrix(log_prob)
        return np.argmax(log_prob,axis=0).tolist()[0]
            
            
#for test

ISOTIMEFORMAT='%Y-%m-%d %X'

def main():
    print('Testing NaiveBayesClassifier...')

    #with open('newtfidf.pkl','rb') as f:
        #x = pickle.load(f)
    #with open('y.pkl','rb') as f:
        #y = pickle.load(f)
    x = getData('newtfidf.pkl')
    y = getData('y.pkl')
    print(x.shape)
    #print(type(y))
    global x1,y1,x2,y2
    x1,x2,y1,y2 = train_test_split(x, y, test_size=0.5, random_state=17)

    #tx1,x2,ty1,y2 = train_test_split(tx2, ty2, test_size=1, random_state=17)

    #print(x1.shape)
    #print(len(y1))
    #print(x2.shape)
    #print(len(y2))

    class_name = list(range(10))
    prior = dict()
    for i in class_name:
        prior[i] = 1 / len(class_name)
        
    if os.path.exists('model.pkl'):
        model = jl.load('model.pkl')
    else:
        print('Fitting ...')
        model = NaiveBayesClassifier(class_name = class_name, prior = prior, adjustPrior = False)
        print(time.strftime( ISOTIMEFORMAT, time.localtime() ))
        model.fit(x1, y1)
        jl.dump(model, 'model.pkl')
    print('Predicting ...')
    print(time.strftime( ISOTIMEFORMAT, time.localtime() ))
    pred = model.predict(x2)
    print('Prediction done')
    print(time.strftime( ISOTIMEFORMAT, time.localtime() ))

    global my, ans
    my = pred
    ans = y2

    #cor = sum([int(pred[i]==y2[i]) for i in range(len(pred))])
    pp, rr = calcPR(pred, y2, class_name)
    print('total test case: ' + str(len(y2)))
    print('P = ' + str(pp))
    print('R = ' + str(rr))
    #print(confusion_matrix(y2, pred))
    print(metrics.classification_report(y2, pred))
    #print('correct predciton: ' + str(cor))
    #print('rate: ' + str(cor / len(pred)))
    print('------------------------')
    with open('pred.pkl','wb') as f:
        pickle.dump(pred,f)


if __name__ == '__main__':
    main()
