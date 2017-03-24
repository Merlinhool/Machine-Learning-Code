import requests
from bs4 import BeautifulSoup
import os
import pickle

def getHref(url, cnt):
    r = requests.get(url)
    #r.encoding = 'gb18030'
    r.encoding = 'UTF-8'

    soup = BeautifulSoup(r.text, "lxml")

    #if cnt == 1: down,up = 36, 75
    #elif cnt < 6: down,up = 37, 76
    #else: down,up = 38,77

    #idx = 0
    href = []
    flag = False
    for i in soup.findAll('a'):
        #if cnt == 6:
            #print(idx)
            #print(i.string)
        #print('-----------------------------------------')
        #print(idx)
        #print(i.string)
        #if idx <= up and idx >= down:
            #href.append((i['href'], i.string))
        if flag:
            href.append((i['href'], i.string))
        if i.string == '每日要闻宏观': flag = True
        if i.string == '上一页': break
        #idx += 1
    return href

def getMoreHref(l, r):
    #if os.path.exists('href.pkl'):
        #with open('href.pkl', 'rb') as f:
            #pre_href = pickle.load(f)
        #tl, tr, href = pre_href
        #if tl == l and tr == r: return f
        
    #print('Getting href list ...')
    href = []
    for cnt in range(l, r):
        #新浪财经
        #url = 'http://roll.finance.sina.com.cn/finance/gjcj/gjjj/index_' + str(cnt) + '.shtml'
        url = 'http://economy.caijing.com.cn/economynews/' + str(cnt) + '.shtml'
        res = getHref(url, cnt)
        #print('-------------------------')
        #print(cnt)
        for i in res:
            #print(i)
            href.append(i)

    #tt = [l, r, href]
    #with open('href.pkl', 'wb') as f:
        #pickle.dump(tt, f)
    return href

essay_count = 0

def getE(url, title):
    global essay_count
    try:
        r = requests.get(url)
        r.encoding = 'UTF-8'
        soup = BeautifulSoup(r.text, "lxml")
        s = soup.get_text()
        res = ""
        n = len(s)
        for i in range(n):
            if s[i:i+3] == '字号：':
                end = i+4
                for j in range(i+3, n):
                    if s[j:j+4] == '(编辑：':
                        end = j
                res = s[(i+3):end]
                break

        #print('-----count:' + str(essay_count))
        #print(title)
        lines = res.splitlines()
        for line in lines:
            if len(line) > 50:
                print(line)
                essay_count += 1
        #print(res)

        
        #soup = BeautifulSoup(r.text, "html.parser")
        #print('--------------------------------------------------')
        #print(soup.prettify())
        #print('--------------------------------------------------')
        #print(url)
        #print(title)
        #print(soup.body)
        #print(soup.get_text())
        #s = soup.get_text()
        #print(type(s))
        #print('--------------------------------------------------')
        return True
    except:
        return False

def getEssays(L):
    cnt = 0
    for i in L:
        tc = 0
        flag = False
        while not flag and tc < 10:
            flag = getE(i[0], i[1])
            tc += 1
        #if cnt >= 0: return
        #cnt += 1

def main():
    href = getMoreHref(1, 51)
    #for i in href: print(i)

    getEssays(href)
    global essay_count
    print('total :' + str(essay_count))
    

if __name__ == '__main__':
    main()
