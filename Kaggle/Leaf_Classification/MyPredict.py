import numpy as np
import pandas as pd
import preprocess

# I changed the ids to 0-based, e.g. from [1,1583] to [0,1582]    
# This function can be rewritten using function `average_probs` and `output_to_csv`
def predict(probs, ids, save_path):
    all_ids = set()
    for i in ids:
        all_ids.add(int(i) + 1)

    ans = pd.read_csv('data/sample_submission.csv', dtype={'id':int})
    ans = ans[ans.id == -1]

    for j in all_ids:
        i = int(j)
        data = probs[np.where(ids == (i-1))[0]]
        p = data.sum(axis = 0)
        p = p / data.shape[0]
        #print(p.sum())
        p = np.insert(p, 0, i)
        p = pd.DataFrame([p], columns = ans.columns)
        p[['id']] = p[['id']].astype(int)
        #print(p)
        #print(ans)
        ans = ans.append(p)
        #print(ans)
        #break
    ans.to_csv(save_path, index=False)
    print('Prediction done.')

def predict_version2(probs, ids, save_path, threshold = 0.99):
    all_ids = set()
    for i in ids:
        all_ids.add(int(i) + 1)

    ans = pd.read_csv('data/sample_submission.csv', dtype={'id':int})
    ans = ans[ans.id == -1]

    def act(x):
        if x > 0.9:
            return 1.0
        else:
            return x

    for j in all_ids:
        i = int(j)
        data = probs[np.where(ids == (i-1))[0]]
        p = data.sum(axis = 0)
        p = p / data.shape[0]

        if p.max() > threshold:
            idx = p.argmax()
            p[:] = 0
            p[idx] = 1.0

        p = np.insert(p, 0, i)
        p = pd.DataFrame([p], columns = ans.columns)
        p[['id']] = p[['id']].astype(int)
        #print(p)
        #print(ans)
        ans = ans.append(p)
        #print(ans)
        #break
    ans.to_csv(save_path, index=False)
    print('Prediction done.')

# return average probs and ids
def average_probs(probs, ids, save_path):
    all_ids = set()
    for i in ids:
        all_ids.add(int(i) + 1)

    ret_p = []
    ret_id = []

    for j in all_ids:
        i = int(j)
        data = probs[np.where(ids == (i-1))[0]]
        p = data.sum(axis = 0)
        p = p / data.shape[0]
        ret_p.append(p)
        ret_id.append(i)

    return np.array(ret_p), np.array(ret_id)
    
def output_to_csv(probs, ids, save_path):
    assert(len(probs) == len(ids))

    ans = pd.read_csv('data/sample_submission.csv', dtype={'id':int})
    ans = ans[ans.id == -1]

    for i in range(len(probs)):
        p = probs[i]
        p = np.insert(p, 0, ids[i])
        p = pd.DataFrame([p], columns = ans.columns)
        p[['id']] = p[['id']].astype(int)
        ans = ans.append(p)

    ans.to_csv(save_path, index=False)
    print('output_to_csv done.')
    
