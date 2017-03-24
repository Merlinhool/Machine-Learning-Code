#要处理未登陆词单字成词的情况
#需要分词的文本不能含有'-'符号
#预计是一个能处理N >= 2的N-gram模型，但是现在只能处理N=2...
#can't not use N-gram with N = 1
import math
import numpy as np
import pandas as pd
from sklearn import linear_model
class Trie:
    son = dict()
    is_word = False
    is_pre = False

    def __init__(self, son = dict(), is_w = False, is_p = False):
        self.son = dict()
        self.is_word = is_w
        self.is_pre = is_p

    def clear(self):
        self.son.clear()
        self.is_wrod = False
        self.is_pre = False
        
    def insert(self, w, is_p):
        if w == "":
            self.is_word = True
            self.is_pre = is_p
        else:
            c = w[0]
            if not c in self.son: self.son[c] = Trie()
            self.son[c].insert(w[1:], is_p)

    def find(self, w):
        if w == "":
            return self
        else:
            c = w[0]
            if c in self.son:
                return self.son[c].find(w[1:])
            else:
                return None

    def isWord(self):
        return self.is_word

    def isPre(self):
        return self.is_pre
            
class Ngram:

    K = 5
    non_pre = 'NON_EXISTS_PREFIX'
    #汉字个数85568：http://www.hwjyw.com/resource/content/2012/02/09/23489.shtml
    #15个标点
    all_single_chinese_charater_count = 85568 + 15

    def __init__(self, start = ('S','_S'), end = ('E','_E')):
        self.data = None
        self.N = 2
        self.pre_count = dict()
        self.pre_to_word_count = dict()
        self.pre_to_word_prob = dict()
        self.sentence_pre_prob = dict()
        self.words = dict()
        self.trie_root = Trie()
        self.sen_start = start
        self.sen_end = end
        self.sentence_pre_count = dict()

    def transProb(self, p):
        try:
            return math.log(p, math.e)
        except:
            raise Exception('math.log() function parameter: ' + str(p))
            
        return math.log(p, math.e)

    def transProbInv(self, p):
        return math.pow(math.e, p)

    def probMult(self, a, b):
        return a + b
    
    def probAdd(self, a, b):
        return self.transProb(self.transProbInv(a) + self.transProbInv(b))

    def probDiv(self, a, b):
        return a - b

    def probChmax(self, a, b):
        return a < b
    
    def probMinValue(self):
        return -math.inf

    def makePrefix(self, pre):
        return '-'.join(pre)
    
    #calc self.words
    def getCountAndProb(self):
        self.pre_count = dict()
        self.pre_to_word_count = dict()
        self.pre_to_word_prob = dict()
        self.pre_to_count_prob = dict()
        self.sentence_pre_prob = dict()
        self.words = dict()
        self.trie_root = Trie()
        self.sentence_pre_count = dict()
        self.single_character = set()

        N = self.N
        prob = self.pre_to_word_prob
        count = self.pre_to_word_count
        pre_count = self.pre_count
        count_prob = self.pre_to_count_prob
        words = self.words
        single_character = self.single_character
        r_single_count = 0
        r_word_count = 0

        train_set_word_count = 0

        #calc count
        for ss in self.data:
            ws = [self.sen_start[0]]
            ws = ws + [i[0] for i in ss]
            #for i in range(N-1): ws.append(self.sen_end[0])
            #only 2-gram
            ws.append(self.sen_end[0])
            #print(ws)

            n = len(ws)
            train_set_word_count += n
            if len(ws) >= N-1: 
                pre = self.makePrefix(ws[0:(N-1)])
                if not pre in self.sentence_pre_count:
                    self.sentence_pre_count[pre] = 1
                else:
                    self.sentence_pre_count[pre] = self.sentence_pre_count[pre] + 1
                    
            for i in range(n-N+1):
                if ws[i] in words: words[ws[i]] = words[ws[i]] + 1
                else: words[ws[i]] = 1
                if len(ws[i]) == 1: 
                    r_single_count += 1
                    single_character.add(ws[i])
                else: r_word_count += 1

                pre = self.makePrefix(ws[i:(i+N-1)])
                c = ws[i+N-1]
                if pre in count:
                    t = count[pre]
                    if c in t: t[c] += 1
                    else: t[c] = 1

                    pre_count[pre] += 1
                else:
                    count[pre] = {c:1}
                    pre_count[pre] = 1

            for i in range(n-N+1, n): 
                if ws[i] in words: words[ws[i]] = words[ws[i]] + 1
                else: words[ws[i]] = 1

                if len(ws[i]) == 1: 
                    r_single_count += 1
                    single_character.add(ws[i])
                else: r_word_count += 1

            flag = True
            for i in ws:
                self.trie_root.insert(i, flag)
                flag = False

        #calc prob

        for (pre,cnt) in self.sentence_pre_count.items():
            self.sentence_pre_prob[pre] = self.transProb((cnt + 0.0) / len(self.data))

        single_word_rate = (r_single_count + 0.0) / r_word_count

        N_gram_count = dict()
        for (pre,ws) in count.items():
            times = dict()
            twordn = 0
            tsinglen = 0
            for (name,cnt) in ws.items():
                if cnt in times:
                    times[cnt] = times[cnt] + 1
                else:
                    times[cnt] = 1

                if len(name) == 1: tsinglen += 1
                else: twordn += 1

            x = []
            y = []
            for (a,b) in times.items():
                x.append([math.log(a)])
                y.append(math.log(b))

            x = pd.DataFrame(x)
            y = np.array(y)

            model = linear_model.LinearRegression()
            model.fit(x,y)
            a = float(model.intercept_)
            b = float(model.coef_[0])

            #tn = len(words - set(ws.keys()))
            tn = len(words.keys()) - len(ws.keys())
            wordn = len(words.keys()) - len(single_character) - twordn + 0.0
            singlen = self.all_single_chinese_charater_count - tsinglen + 0.0

            if tn == 0: 
                p_single = self.probMinValue()
                p_word = self.probMinValue()
                #pp0 = self.probMinValue()
            else:
                if 1 in times:
                    pp0 = (times[1] + 0.0) / pre_count[pre]
                else:
                    pp0 = math.pow(math.e, a) / pre_count[pre]

                if wordn == 0:
                    p_single = self.transProb(pp0)
                    p_word = self.probMinValue()
                else:
                    p_single = single_word_rate * pp0 / (1.0 + single_word_rate) / singlen
                    p_single = self.transProb(p_single)
                    p_word = pp0 / (1.0 + single_word_rate) / wordn
                    p_word = self.transProb(p_word)

            prob[pre] = (ws, times, a, b, p_word, p_single)

        times = dict()
        for (name,cnt) in words.items():
            if cnt in times:
                times[cnt] = times[cnt] + 1
            else:
                times[cnt] = 1

        x = []
        y = []
        for (a,b) in times.items():
            x.append([math.log(a)])
            y.append(math.log(b))
            
        x = pd.DataFrame(x)
        y = np.array(y)
        
        model = linear_model.LinearRegression()
        model.fit(x,y)
        a = float(model.intercept_)
        b = float(model.coef_[0])
        unseen_single = self.all_single_chinese_charater_count - len(single_character)
        pre_count[self.non_pre] = train_set_word_count
        if 1 in times:
            pp0 = self.transProb((times[1] + 0.0) / train_set_word_count / unseen_single)
        else:
            pp0 = self.probDiv(self.transProb(math.pow(math.e, a)), 1.0 * train_set_word_count * unseen_single)
        prob[self.non_pre] = (words, times, a, b, self.probMinValue(), pp0)

    #the predict function needs these parameters:
    #self.sentence_pre_prob
    #self.pre_to_word_prob
    #Trie

    def fit(self, data, N = 2):
        self.data = data
        self.N = N

        self.getCountAndProb()

        del self.data

    #Good-Turing Estimation
    def calc_prob(self, word, pre):
        ws, times, a, b, p_word, p_single = self.pre_to_word_prob[pre]
        N = self.pre_count[pre]
        if not word in ws: 
            if len(word) == 1: return p_single
            else: return p_word
        else:
            #then 'cnt in times = True'
            cnt = ws[word]
            Nc = times[cnt] + 0.0
            if cnt+1 in times: Nc1 = times[cnt+1] + 0.0
            else: Nc1 = math.pow(math.e, a + b * math.log(cnt+1))
            if self.K + 1 in times: Nk1 = times[self.K+1] + 0.0
            else: Nk1 = math.pow(math.e, a + b * math.log(self.K+1))
            if 1 in times: N1 = times[1] + 0.0
            else: N1 = math.pow(math.e, a)
            c_star = (cnt + 1) * Nc1 / Nc

                #if max(times.keys()) <= self.K:
                    #c_star = (cnt + 1) * Nc1 / Nc
                #else:
                    #c_star = ( ((cnt+1) * Nc1 / Nc) - cnt * (self.K+1.0) * Nk1 / N1 ) / (1.0 - (self.K+1.0) * Nk1 / N1)

                #if c_star < 0:
                    #print('--------------------------------------------------------')
                    #print(cnt)
                    #print(Nc)
                    #print(Nc1)
                    #print(Nk1)
                    #print(N1)
                    #print(self.K)
                    #print(a)
                    #print(b)
                    #print('-----')
                    #for i in times.items():
                        #tmp = math.pow(math.e, a + b * math.log(i[0]))
                        #print(str(i) + "  " + str(tmp))
                        
            return self.probDiv(self.transProb(c_star), self.transProb(N + 0.0))

    def find_first_pre(self):
        node = self.trie_root
        pos = 0
        res = []
        res_str = ""
        for c in self.pred_sentence:
            pos = pos + 1
            res_str = res_str + c
            node = node.find(c)
            if node == None: break
            if node.isPre(): res.append((res_str, pos))

        return res
    
    def find_new_pre(self, pre, idx):
        if self.N > 2:
            pos = pre.find('-')
            npre = pre[(pos+1):] + '-'
        else:
            npre = ""

        tmp = self.pred_sentence[idx:]
        node = self.trie_root
        res = []
        word = ""
        prob = self.pre_to_word_prob
        flag = True
        for c in tmp:
            npre = npre + c
            word = word + c
            idx = idx + 1
            node = node.find(c)
            if flag:
                flag = False
                if node == None or not node.isWord():
                    res.append((idx, self.non_pre, self.pre_to_word_prob[pre][5]))
                                
            if node == None: break
            #if node.isWord() and (word in prob[pre]):
                #res.append((idx, npre, prob[pre][word]))
            if node.isWord():
                t = self.calc_prob(word, pre)
                if t > self.probMinValue(): res.append((idx, npre, t))
        return res

    def predict_sentence(self, s):
        self.pred_sentence = s
        dp = []
        path_back = []
        #non_pre_path_back = []
        for i in range(len(s) + 1):
            dp.append(dict())
            path_back.append(dict())
            #non_pre_path_back.append(dict())
        arr = self.find_first_pre()
        for (pre, idx) in arr:
            dp[idx][pre] = self.sentence_pre_prob[pre]
            path_back[idx][pre] = ""
            #if pre == self.non_pre: non_pre_path_back[idx][pre] = ""

        for i in range(len(s)):
            for (pre,ipr) in dp[i].items():
                new_pre = self.find_new_pre(pre, i)
                for (it, npre, pr) in new_pre:
                    tp = self.probMult(ipr, pr)
                    if not ((npre in dp[it]) and (not self.probChmax(dp[it][npre], tp))):
                        dp[it][npre] = tp
                        path_back[it][npre] = pre
        
        mx = 0
        mx_pre = None
        for (pre, p) in dp[len(s)].items():
            if mx_pre == None or self.probChmax(mx, p):
                mx = p
                mx_pre = pre

        if mx_pre == None:
            raise Exception('**无法切分**')
            return ["**无法切分**"], 0

        it = len(s)
        res = []
        while mx_pre != "":
            res.append(mx_pre)
            if mx_pre == self.non_pre:
                mx_pre, it = path_back[it][mx_pre], it - 1
            else:
                mx_pre, it = path_back[it][mx_pre], it - len(mx_pre)

        res.reverse()
        return res, mx

    #e.g. yes, I do.
    #sep = None, return ["yes", ",", "I", "do"]
    #sep = '/', return "yes/,/I/do/."
    def predict(self, testset, sep = None):
        res = []
        for s in testset:
            t,p = self.predict_sentence(self.sen_start[0] + s + self.sen_end[0])

            t = t[1:-1]

            ts = 0
            for i in range(len(t)):
                if t[i] == self.non_pre:
                    t[i] = s[ts]
                    ts += 1
                else:
                    ts += len(t[i])

            flag = 0
            while flag != None:
                idx = flag
                flag = None
                for i in range(idx, len(t)):
                    if len(t[i]) != 1 or t[i] in self.words: continue
                    j = i + 1
                    new_word = t[i]
                    while j < len(t):
                        if len(t[j]) != 1 or t[j] in self.words: break
                        new_word += t[j]
                        j += 1
                    t = t[:i] + [new_word] + t[j:]
                    flag = j
                    break
            
            if sep != None: 
                res.append(sep.join(t))
            else:
                res.append(t)
        return res

if __name__ == '__main__':
    print('Running SimpleNgram.py')
