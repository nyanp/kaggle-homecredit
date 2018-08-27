import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

base_models = {
    'lgbm_base':['output%2F{}_lgbm_seed0','output%2F{}_lgbm_seed1','output%2F{}_lgbm_seed2','output%2F{}_lgbm_seed3','output%2F{}_lgbm_seed4','output%2F{}_lgbm_seed5'],
    'xgb':['output%2F{}_xgb_seed0','output%2F{}_xgb_seed1','output%2F{}_xgb_seed2'],
    'full':['180821/output%2F{}_full'],
    'app': ['180821/output%2F{}_app'],
    'app_prev': ['180821/output%2F{}_app_prev'],
    'app_ins': ['180821/output%2F{}_app_ins'],
    'app_credit': ['180821/output%2F{}_app_credit'],
    'app_pos': ['180821/output%2F{}_app_pos'],
    'lgb_p3': ['{}_lgb46_p3']

    #'nn0',
    #'nn0_seed2',
    #'nn2_seed2'
}

base_path = '../stack/'

df = pd.read_feather('../input/application_all.f')
y_train = df[~df.TARGET.isnull() & (df['CODE_GENDER'] != 'XNA')].TARGET

def load_1st_model(name, basepath='', y_train=None):
    train_oof = np.load(basepath+name.format('train')+'.npy')
    test = np.load(basepath+name.format('test')+'.npy')

    if y_train is not None:
        print('len: {}'.format(len(train_oof)))
        print("{}: {}".format(name, roc_auc_score(y_train, train_oof)))

    return train_oof, test

def load_1st_model_rank_averaging(names, basepath='', y_train=None):
    trains = []
    tests = []
    for n in names:
        tr, tt = load_1st_model(n, basepath, None)

        dtr = pd.DataFrame()
        dtr['target'] = tr
        dtr['target'] = dtr['target'].rank() / len(dtr)

        dtt = pd.DataFrame()
        dtt['target'] = tt
        dtt['target'] = dtt['target'].rank() / len(dtt)

        trains.append(dtr)
        tests.append(dtt)

    train_oof = pd.concat(trains, axis=1).mean(axis=1)
    test = pd.concat(tests, axis=1).mean(axis=1)

    if y_train is not None:
        print('len: {}'.format(len(train_oof)))
        print("{}: {}".format(names, roc_auc_score(y_train, train_oof)))

    return train_oof.values, test.values


def load(name, basepath='', y_train=None):
    if isinstance(name, str):
        return load_1st_model(name, basepath, y_train)
    else:
        return load_1st_model_rank_averaging(name, basepath, y_train)


files = [load(base_models[m], base_path, y_train) for m in base_models]

x_train = np.transpose(np.vstack(tuple([x[0] for x in files])))
x_test = np.transpose(np.vstack(tuple([x[1] for x in files])))

parameters = {
    'penalty': ('l1','l2'),
    #'C': (0.01,0.1,1,10)
    'C': (0.001,0.003,0.01,0.03)
}

print('start cv...')

cv = GridSearchCV(LogisticRegression(), parameters, verbose=3, scoring='roc_auc', cv=5)
cv.fit(x_train, y_train)

print('best: {}, params: {}'.format(cv.best_score_, cv.best_params_))

cv.best_estimator_.fit(x_train, y_train)
print(cv.best_estimator_.coef_)


test = pd.DataFrame()

preds = cv.best_estimator_.predict_proba(x_test)[:,0]
test['TARGET'] = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))

test['SK_ID_CURR'] = df[df.TARGET.isnull()].reset_index()['SK_ID_CURR']

print(test.head(20))
print(test.describe())
print(test.shape)

test.to_csv('test_stack_180828.csv',index=False)
