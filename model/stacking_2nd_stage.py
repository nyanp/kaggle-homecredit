import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

base_models = [
    'full',
    'app',
    'app_prev',
    'app_ins',
    'app_pos',
    'app_bureau',
    'app_credit',
    #'nn0',
    #'nn0_seed2',
    #'nn2_seed2'
]

base_path = '../stack/180821/output%2F'

df = pd.read_feather('../input/application_all.f')
y_train = df[~df.TARGET.isnull() & (df['CODE_GENDER'] != 'XNA')].TARGET

def load_1st_model(name, basepath='', y_train=None):
    train_oof = np.load(basepath+'train_'+name+'.npy')
    test = np.load(basepath+'test_'+name+'.npy')

    if y_train is not None:
        print('len: {}'.format(len(train_oof)))
        print("{}: {}".format(name, roc_auc_score(y_train, train_oof)))

    return train_oof, test

files = [load_1st_model(m, base_path, y_train) for m in base_models]

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

test.to_csv('test_stack_ridge2.csv',index=False)
