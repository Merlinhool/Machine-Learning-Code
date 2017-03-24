import random
import string
import SimpleNgram
import pickle
import os

def Input(path): 
    sentence = []
    data = []
    with open(path, 'r') as f:
        for line in f:
            while line != "" and (line[-1] == '\n' or line[-1] == ' '): 
                line = line[:-1]
            if line == "": continue

            line = line[23:]
            line = line.split('  ')
            elem = []
            #elem = [('S','_S')]
            for s in line:
                if s[0] == '[':
                    s = s[1:]
                pos = s.find('/')
                if pos == -1: raise Exception('wrong split:' + s)

                e = (s[:pos], s[(pos+1):])
                for c in e[1]:
                    if string.ascii_letters.find(c) == -1:
                        if c == ']' or c == '/':
                            e = (e[0], e[1][:e[1].find(']')])
                            break
                        else: raise Exception('wrong tag:' + e[1])

                elem.append(e)

            #elem.append(('E', '_E'))
            data.append(elem)

    return data

#permutation
def perm(x, n):
    #return x
    sam = random.sample(range(len(x)), n)
    return [x[i] for i in sam]

def generateThreeSet(sentence, r1, r2):
    n = len(sentence)
    cnt1 = int(n * r1)
    cnt2 = int(n * r2) + cnt1
    data = perm(sentence, len(sentence))
    train = data[:cnt1]
    devset = data[cnt1:cnt2]
    test = data[cnt2:]
    return train, devset, test

def toSentence(sen):
    res = []
    for l in sen:
        s = ''
        for pr in l:
            s = s + pr[0]
        res.append(s)
    return res

#train : validate : test = trainRatio : devsetRation : (1 - trainRation - devsetRatio)
trainRatio = 0.8
devsetRatio = 0
#'0.8 : 0.1 : 0.1' is from the textbook, $4.3 Training and Test Sets

def getFirst(s):
    res = []
    for i in s:
        ans = []
        for j in i:
            ans.append(j[0])
        res.append(ans)
    return res

def getDataAndModel():
    cache_path = 'cache.pkl'
    if os.path.exists(cache_path):
        print('Loading cache...')
        with open(cache_path, 'rb') as f:
            cac = pickle.load(f)
        #train, dev, devset, test, testset, model = cac
        return cac
    else:
        path = 'YL.md'
        data = Input(path)
        (train, devset, testset) = generateThreeSet(data, trainRatio, devsetRatio)
        dev = toSentence(devset)
        test = toSentence(testset)
        testset = getFirst(testset)
        devset = getFirst(devset)
         
        print('Training model...')
        model = SimpleNgram.Ngram(start = ('S','_S'), end = ('E','_E'))
        model.fit(train, N = 2)
        #model.debug()
        cac = (train, dev, devset, test, testset, model)
        with open('cache.pkl', 'wb') as f:
            pickle.dump(cac, f)
        return cac

def predict(model, test):
    print('Predict...')
    return model.predict(test)

def getSet(p):
    start = end = 0
    ans = set()
    for i in p:
        end = start + len(i)
        ans.add((start, end))
        start = end
    return ans

def analyse(pred, test):
    print('Comparing with answer...') 
    cor_sum, my_sum, ans_sum = 0, 0, 0
    for i in range(len(pred)):
        p = pred[i]
        ans = test[i]
        #if i < 3:
            #print(p)
            #print(ans)
        set1 = getSet(p)
        set2 = getSet(ans)

        cor_sum += len(set1 & set2)
        my_sum += len(p)
        ans_sum += len(ans)

    p0 = (cor_sum + 0.0) / my_sum
    r0 = (cor_sum + 0.0) / ans_sum
    F1 = 2 * p0 * r0 / (p0 + r0)
    print('P = ' + str(p0))
    print('R = ' + str(r0))
    print('F1 = ' + str(F1))

    #with open('res.txt', 'w') as f:
        #f.write('P = ' + str(p0) + '\n')
        #f.write('R = ' + str(r0) + '\n')
        #f.write('F1 = ' + str(F1) + '\n')
        #for i in pred:
            #f.write(' / '.join(i) + '\n')


if __name__ == '__main__': 
    train, dev, devset, test, testset, model = getDataAndModel()
    pred = predict(model, test)
    analyse(pred, testset)
