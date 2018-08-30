import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


def load_1st_model(name, basepath=''):
    test = np.load(basepath + name.format('test') + '.npy')

    return test


def parsenames(names):
    if isinstance(names, list):
        return names
    else:
        n = int(names['n'])
        return [names['base'].format(i) for i in range(n)]


def load_1st_model_rank_averaging(names, basepath=''):

    tests = []
    for n in parsenames(names):
        tt = load_1st_model(n, basepath)

        dtt = pd.DataFrame()
        dtt['target'] = tt
        dtt['target'] = dtt['target'].rank() / len(dtt)

        tests.append(dtt)

    test = pd.concat(tests, axis=1).mean(axis=1)

    return test


base_models2 = {
    'lgbm_base':{'base': 'undersample/{{}}_lgbm_sample8000_seed{}', 'n':20, 'w':3},
    'xgb':{'base': 'undersample/{{}}_xgb_seed{}_uc', 'n':2, 'w':2},
    'app': {'base': 'undersample/{{}}_app_seed{}_uc', 'n':10, 'w':1},
    'lgb_p3': ['{}_lgb46_p3','{}_lgb46_p4','{}_lgb46_p5','{}_lgb46_p6','{}_lgb46_p7'],
    'nn': {'base': '{{}}_nn_normal_v0_seed{}', 'n':10, 'w':1},
    #'neptune': {'kernel':1, 'base':'../output/neptune.csv.csv', 'w':2},
    #'blend': {'kernel': 1, 'base': '../output/Submission_HomeCredit_Blend.csv', 'w': 0.5},
}

dfs = []
for d in base_models2:
    n = base_models2[d]

    if 'kernel' in n:
        df = pd.read_csv(n['base'])['TARGET']
        df = df.rank() * n['w'] / len(df)
    elif 'w' in n:
        df = load_1st_model_rank_averaging(n, '../stack/') * n['w']
    else:
        df = load_1st_model_rank_averaging(n, '../stack/')
    dfs.append(df)

df = pd.concat(dfs, axis=1).sum(axis=1)

df = (df - df.min()) / (df.max() - df.min())

base = pd.read_csv('../input/sample_submission.csv')

assert len(df) == len(base)

base['TARGET'] = df

print(base.shape)
print(base.head())

base.to_csv('../output/rank_averaging_180829_uc.csv',index=False)