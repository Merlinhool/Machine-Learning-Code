import os
import numpy as np
import pandas as pd
import pickle
import sys

from scipy import ndimage

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#from skimage import transform
#from skimage import io
#from skimage import img_as_float
import PIL
from PIL import Image

import matplotlib.pyplot as plt

N = 1584
SIZE = 128

# delete rows of all zero and columns of all zero
def zeropad(pic, height, width):
    diff = abs(height - width)
    mx = max(height, width)
    if height > width:
        pic = np.concatenate((
            np.zeros((mx, diff // 2), dtype=int),
            pic,
            np.zeros((mx, diff - diff // 2), dtype=int)
        ), axis=1)
    elif height < width:
        pic = np.concatenate((
            np.zeros((diff // 2, mx), dtype=int),
            pic,
            np.zeros((diff - diff // 2, mx), dtype=int)
        ), axis=0)
    return pic, (mx,mx)

def clip(pic):
    x, y = np.where(pic > 0)
    print(x.min(), x.max())
    print(y.min(), y.max())
    lx = max(x.min()-5, 0)
    rx = min(x.max()+5, pic.shape[0])
    ly = max(y.min()-5, 0)
    ry = min(y.max()+5, pic.shape[1])
    return pic[lx:rx, ly:ry]

def image_preprocess():
    os.chdir('data/images')
    pics = []
    for i in range(1,N+1):
        if i%100==0: print(i)
        pics.append([])
        tmp = pics[-1]
        pic = Image.open(str(i) + '.jpg')
        data = np.array(pic.getdata()).reshape((pic.height, pic.width))
        data,shape = zeropad(data, pic.height, pic.width)
        pic = Image.new('L', shape)
        pic.putdata(data.flatten())
        pic = pic.resize((SIZE,SIZE), resample=PIL.Image.HAMMING)

        for j in range(4):
            P = np.array(pic.getdata(), dtype=np.float32).reshape(SIZE,SIZE)
            tmp.append(P)
            P = np.array(pic.transpose(PIL.Image.FLIP_TOP_BOTTOM).getdata(), dtype=np.float32).reshape((SIZE,SIZE))
            tmp.append(P)
            pic = pic.rotate(angle=90, resample=PIL.Image.BICUBIC, expand=True)
            if pic.size != (SIZE,SIZE):
                pic = pic.resize((SIZE,SIZE))

    os.chdir('..')
    os.chdir('..')
    #for j in pics:
        #for i in j:
            #print(i.dtype, i.min(), i.max())
    return pics

pics_mean = None
pics_std = None
fea_mean = None
fea_std = None

def load_data(path, flag):
    print('Loading ' + flag + ' data...')
    data = pd.read_csv(path)
    ids = data.pop('id') - 1
    data = data.values
    if flag == 'train':
        label_encoder = preprocessing.LabelEncoder()
        y = data[:,0]
        label_encoder.fit(y)
        y = label_encoder.transform(y)

        x = data[:,1:]
        x = x.astype(np.float32)
        x = x - x.mean(axis=0)
        index = np.arange(data.shape[0])
        xt, xv, yt, yv = train_test_split(index, y, test_size=0.1, random_state=0)
        train_pics = []
        train_fea = []
        train_y = []
        val_pics = []
        val_fea = []
        val_y = []
        it = -1
        for i in xt:
            it += 1
            for j in pics[ids[i]]:
                train_pics.append(j)
                train_fea.append(x[i,:])
                train_y.append(yt[it])

        it = -1
        for i in xv:
            it += 1
            for j in pics[ids[i]]:
                val_pics.append(j)
                val_fea.append(x[i,:])
                val_y.append(yv[it])

        n = len(train_pics)
        m = len(val_pics)
        train_pics = np.array(train_pics).reshape(n, 1, SIZE, SIZE)
        train_fea = np.array(train_fea).reshape(n, 3, 8, 8)
        val_pics = np.array(val_pics).reshape(m, 1, SIZE, SIZE)
        val_fea = np.array(val_fea).reshape(m, 3, 8, 8)
        train_y = np.array(train_y)
        val_y = np.array(val_y)

        #print(train_pics.shape)
        #train_pics = train_pics[:30, :]
        #print(train_pics.shape)
        #print(train_fea.shape)
        #train_fea = train_fea[:30, :]
        #print(train_fea.shape)
        #print(train_y.shape)
        #train_y = train_y[:30]
        #print(train_y.shape)

        print(train_pics.dtype)
        #print(train_fea.dtype)
        #print(val_pics.dtype)
        #print(val_fea.dtype)

        global pics_mean, pics_std, fea_mean, fea_std
        pics_mean = np.mean(train_pics, axis=0)
        #pics_std = np.std(train_pics, axis=0)
        #pics_std = ((train_pics - pics_mean)**2).sum(axis=0) / (train_pics.shape[0]+0.0)
        fea_mean = np.mean(train_fea, axis=0)
        #t = np.std(train_fea,axis=0)
        fea_std = np.std(train_fea, axis=0)
        #fea_std = ((train_fea-fea_mean)**2).sum(axis=0) / (train_fea.shape[0]+0.0)

        #print(train_pics[np.where(train_pics[0]>0)[0]][0])
        train_pics = np.round(train_pics / 255.0)
        #train_pics /= 255.0
        #train_pics -= pics_mean
        train_fea -= fea_mean
        train_fea /= fea_std

        val_pics = np.round(val_pics / 255.0)
        #val_pics /= 255.0
        #val_pics -= pics_mean
        val_fea -= fea_mean
        val_fea /= fea_std

        return train_pics, train_fea, train_y, val_pics, val_fea, val_y, label_encoder
    else:
        x = data
        x = x - x.mean(axis=0)
        test_pics = []
        test_fea = []
        test_id = []

        for it in range(len(x)):
            for j in pics[ids[it]]:
                test_pics.append(j)
                test_fea.append(x[it])
                test_id.append(ids[it])

        n = len(test_pics)
        test_pics = np.array(test_pics).reshape(n, 1, SIZE, SIZE)
        test_fea = np.array(test_fea).reshape(n, 3, 8, 8)
        test_id = np.array(test_id)

        test_pics = np.round(test_pics / 255.0)
        #test_pics /= 255.0
        #test_pics -= pics_mean
        #test_pics /= pics_std
        test_fea -= fea_mean
        test_fea /= fea_std
        return test_pics, test_fea, test_id

pics_path = 'pics.pkl'
if os.path.exists(pics_path):
    pass
    with open(pics_path,'rb') as f:
        pics = pickle.load(f)
else:
    pics = image_preprocess()
    with open(pics_path, 'wb') as f:
        pickle.dump(pics, f)

print('preprocessing images done.')

def main():
    tr_p, tr_f, tr_y, va_p, va_f, va_y, label_encoder = load_data('data/train.csv', 'train')
    te_p, te_f, te_id = load_data('data/test.csv', 'test')

if __name__ == '__main__':
    main()
