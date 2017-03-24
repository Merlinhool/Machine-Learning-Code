import jieba
import copy
import pickle
import re
import os
import numpy as np
import pandas as pd
import jieba.analyse
from lda import LDA
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model  import LogisticRegression
from sklearn.externals import joblib
from scipy.sparse import bmat
from sklearn.ensemble import GradientBoostingClassifier



labelList = ['汽车时代','旅游休闲','经济论坛','情感天地','游戏地带','体育聚焦','国际观察','我的大学','ＩＴ视界','影视评论']
otherStopWords = set([' '])
#global allStopWords

def Dump(s, name, flag=None):
    with open(name+'.pkl', 'wb') as f: 
        if flag == 'joblib':
            joblib.dump(s, f)
        else:
            pickle.dump(s, f)

def Load(name, flag=None):
    with open(name+'.pkl', 'rb') as f:
        if flag == 'joblib':
            data = joblib.load(f)
        else:
            data = pickle.load(f)
    return data

def getAllStopWords():
    stopWords = set()
    with open('allStopWords.txt', 'rb') as f:
        for line in f.readlines():
            line = line.decode('utf-8')
            if line[-1] == '\n': line = line[:-1]
            if line == '': continue
            #print(line + ',')
            stopWords.add(line)
    stopWords |= otherStopWords
    with open('allStopWords.txt','w') as f:
        for i in stopWords:
            f.write(i + '\n')

    #print('stopWords:')
    #for i in stopWords: print(i)
    global allStopWords
    allStopWords = stopWords

def removeStopWord(label):
    global allStopWords
    stopWords = allStopWords

    text = []
    for name in label:
        data = open(name + '1.txt', 'rb')
        output = open(name + '_stop_words_removed.txt', 'w')
        out = open(name + '_digit_and_tab_removed.txt', 'w')
        i = 0
        for line in data.readlines():
            try:
                s = line.strip().decode('utf-8')
                s = re.sub(r'[0-9]+', '', s)
                s = re.sub('\t',' ',s)
                out.write(s + '\n')
                cut = jieba.cut(s)
                removed = ' '.join(list(set(cut) - stopWords))

                text.append(removed)
                output.write(removed + '\n')
                i += 1
                if i > 19999: 
                    break
            except:
                global lin
                print(name)
                print(i)
                lin = line
        if i <= 19999:
            raise Exception('the amount of ' + str(name) + ' is ' + str(i)) 
        data.close()
        out.close()
        output.close()
    return text

def getAllText(label = labelList):
    fp = 'allText'
    if os.path.exists(fp + '.pkl'):
        print('Load allText...')
        text = Load(fp)
    else:
        print('cut text and remove stop words...')
        text = removeStopWord(label)
        Dump(text, fp)
    return text

def _getWordDict(label,topK):
    global allStopWords
    stopWords = allStopWords
    jieba.analyse.set_stop_words('./allStopWords.txt')

    wordDict = dict()
    idx = 0
    output = open('wordDict.txt', 'w')
    for name in label:
        print(name)
        data = open(name + '_digit_and_tab_removed.txt', 'r')
        tags = jieba.analyse.extract_tags(data.read(), topK=topK, withWeight=True)
        i = 0
        for tag in tags:
            s = str(i) + '\t\t' + name + '\t\t' + str(tag[0]) + '\t\t' + str(tag[1])
            output.write(s + '\n')
            if not (tag[0] in wordDict):
                wordDict[tag[0]] = idx
                idx += 1
            i += 1
        data.close()
    Dump(wordDict, 'wordDict')
    output.close()
    return wordDict

def getWordDict(label = labelList):
    fp = 'wordDict'
    topK = 10000
    if os.path.exists(fp + '.pkl'):
        print('Load wordDict...')
        wordDict = Load(fp)
    else:
        print('generate wordDict...')
        wordDict = _getWordDict(label, topK)
        Dump(wordDict, fp)
    return wordDict

def _getAllText(label):
    text = []
    for name in label:
        data = open(name + '3.txt','r')
        i = 0
        for line in data.readlines():
            text.append(line.strip())
            i += 1
            if i > 19999:
                break
    return text

def getAllText(label = labelList):
    path = 'allText'
    if os.path.exists(path + '.pkl'):
        print('Load allText from pickle...')
        allText = Load(path)
    else:
        print('input allText...')
        allText = _getAllText(label)
        Dump(allText, path)
    return allText
    

def _getTfidfMatrix(text, label):
    vec = TfidfVectorizer()
    x_train = vec.fit_transform(text)
    class_num = len(label)
    each_num = int(x_train.shape[0] / class_num)
    if (x_train.shape[0] % class_num) != 0:
        raise Exception('x_tarin[0] % class_num = ' + str(x_train.shape[0] % class_num))
    y_train = []
    for i in range(class_num):
        for j in range(each_num):
            y_train.append(i)
    print('x_train.shape: ' + str(x_train.shape))
    print('y_train.shape: ' + str(len(y_train)))
    Dump(x_train, 'x_all_tfidf')

    x_train_new = SelectKBest(chi2, k=5000).fit_transform(x_train, y_train)
    print('new x_train.shape: ' + str(x_train_new.shape))
    return x_train_new, y_train

def getTfidfMatrix(text, label = labelList):
    if os.path.exists('x_tfidf.pkl') and os.path.exists('y.pkl'):
        print('Load tfidf (x and y)...')
        x = Load('x_tfidf')
        y = Load('y')
    else:
        print('generate tfidf...')
        x,y = _getTfidfMatrix(text, label)
        Dump(x, 'x_tfidf')
        Dump(y, 'y')
    return x, y

def _getLDA(text, label, n_topic_words):
    vectorizer = CountVectorizer(min_df = 100, max_df = 5000)
    transformer = TfidfTransformer()
    df = vectorizer.fit_transform(text)
    tfidf_word_name = vectorizer.get_feature_names()

    model = LDA(n_topics=20, n_iter=1000, random_state=1)
    model.fit(df)
    Dump(model, 'LDA_model', 'joblib')
    topic_word = model.topic_word_
    doc_topic = model.doc_topic_
    with open('topic_word.txt','w') as f:
        n_top_words = 300
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(tfidf_word_name)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            f.write('Topic {}: {}'.format(i, ' '.join(topic_words)) + '\n')
    return topic_word, doc_topic

def getLDA(text, label = labelList):
    fp1 = 'topic_word'
    fp2 = 'doc_topic'
    if os.path.exists(fp1 + '.pkl') and os.path.exists(fp2 + '.pkl'):
        print('Load topic_word and doc_topic...')
        t1 = Load(fp1)
        t2 = Load(fp2)
    else:
        print('generate ' + fp1 + ' and ' + fp2 + '...')
        t1,t2 = _getLDA(text, label, n_topic_words = 300)
        Dump(t1, fp1)
        Dump(t2, fp2)
    return t1, t2

def mergeX(x1, x2, y):
    print('x1.shape' + str(x1.shape))
    print('x2.shape' + str(x2.shape))
    print('y.shape' + str(len(y)))
    return bmat([[x1, x2]])

def result_analyse(pred, ans, path):
    global label_name
    report = metrics.classification_report(ans, pred, target_names = label_name)
    #confusion_mtx = metrics.confusion_matrix(ans, pred)
    confusion_mtx = metrics.confusion_matrix(ans, pred)
    index = [(str(i[0]) + '-' + str(i[1])) for i in list(zip(range(len(label_name)), label_name))]
    mtx = pd.DataFrame(confusion_mtx, index = index)
    #global mtx
    #mtx = confusion_mtx
    #print(confusion_mtx)
    with open(path+'_result_report.txt', 'w') as f:
        f.write('classfication report\n')
        f.write(report)
        f.write('\n\n\n')
        f.write('confusion matrix\n')
        f.write(str(mtx))
        f.write('\n\n\n')

def predict_LR_tfidf(x, y, testLabel):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5, random_state=17)
    model = LogisticRegression()
    print('train model...')
    print('x_shape: ' + str(x_train.shape))
    model.fit(x_train, y_train)
    Dump(model, 'LR_tfidf', 'joblib')
    print('predict...')
    print('x_shape: ' + str(x_test.shape))
    pred = model.predict(x_test)
    Dump(pred, 'LR_tfidf_prediction')
    result_analyse(pred, y_test, 'LR_tfidf')
    
def predict_LR_tfidf_lda(x, y, testLabel):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5, random_state=17)
    model = LogisticRegression()
    print('train model...')
    print('x_shape: ' + str(x_train.shape))
    model.fit(x_train, y_train)
    Dump(model, 'LR_tfidf_lda', 'joblib')
    print('predict...')
    print('x_shape: ' + str(x_test.shape))
    pred = model.predict(x_test)
    Dump(pred, 'LR_tfidf_lda_prediction')
    result_analyse(pred, y_test, 'LR_tfidf_lda')

def predict_LR_tfidf_lda_gdbt(x, y, model_path, testLabel):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5, random_state=17)
    path = 'gbdt'
    if os.path.exists(path + '.pkl'):
        print('Load gbdt...')
        gbdt = Load(path, 'joblib')
    else: 
        print('fit gbdt...')
        gbdt = GradientBoostingClassifier(max_depth=5)
        gbdt.fit(x_train, y_train)
        Dump(gbdt, path, 'joblib')
    
    path = 'oneHot'
    if os.path.exists(path + '.pkl'):
        print('Load oneHot...')
        oneHot = Load(path, 'joblib')
    else: 
        print('fit oneHot...')
        oneHot = preprocessing.OneHotEncoder()
        oneHot.fit(apply_data[:, :, 0])
        Dump(oneHot, path, 'joblib')

    print('transform x_train and x_test')
    apply_data = gbdt.apply(x_train)
    gbdt_features = oneHot.transform(apply_data[:, :, 0])
    x_train = bmat([[x_train, gbdt_features]])

    apply_data = gbdt.apply(x_test)
    gbdt_features = oneHot.transform(apply_data[:, :, 0])
    x_test = bmat([[x_test, gbdt_features]])

    Dump((x_train, x_test, y_train, y_test), 'splitted_data')
    print('new x_train.shape' + str(x_train.shape))

    ################train model and predict#################
    if os.path.exists(model_path):
        print('Load model...')
        model = Load(model_path, 'joblib')
    else:
        if model_path == 'LogisticRegression': 
            model = LogisticRegression()
        elif model_path == 'SVM_linear':
            model = svm.SVC(kernel='linear', C=1)
        elif model_path == 'GaussianNB':
            model = GaussianNB()
            x_train = x_train.todense()
            x_test = x_test.todense()
        else:
            raise Exception('unexpected model name: ' + model_path)
        print('train model...')
        print('x_shape: ' + str(x_train.shape))
        model.fit(x_train, y_train)
        Dump(model, model_path, 'joblib')
        
    print('predict...')
    print('x_shape: ' + str(x_test.shape))
    pred = model.predict(x_test)
    Dump(pred, 'LR_tfidf_lda_prediction')
    result_analyse(pred, y_test, model_path)

def main():
    #testLabel = ['汽车时代','旅游休闲','经济论坛','情感天地','体育聚焦']
    testLabel = labelList
    getAllStopWords()
    #global allText
    #allText = getAllText(testLabel)
    #wordDict = getWordDict(testLabel)

    
    #del some items from the data set...
    #get allText 
    allText = getAllText(testLabel)
    x_tfidf,y = getTfidfMatrix(allText, testLabel)
    topic_word,doc_topic = getLDA(allText, testLabel)

    global x_tfidf_g, doc_topic_g
    x_tfidf_g = x_tfidf
    doc_topic_g = doc_topic

    x_tfidf_lda = mergeX(x_tfidf, doc_topic, y)

    print('x_tfidf_lda.shape:')
    print(x_tfidf_lda.shape)

    ####################predict###############################
    global label_name
    label_name = testLabel
    #LR, tfidf
    #predict_LR_tfidf(x_tfidf, y, testLabel)

    #LR, tfidf, LDA
    #predict_LR_tfidf_lda(x_tfidf_lda, y, testLabel)

    #LR or SVM, tfidf, LDA, GDBT
    #predict_LR_tfidf_lda_gdbt(x_tfidf_lda, y, 'GaussianNB', testLabel)
    #predict_LR_tfidf_lda_gdbt(x_tfidf_lda, y, 'LogisticRegression', testLabel)
    predict_LR_tfidf_lda_gdbt(x_tfidf_lda, y, 'SVM_linear', testLabel)


if __name__=='__main__':
    main()
