import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

base_models = {
    'lgbm_base':{'base': '{{}}_lgbm_seed{}', 'n':20},
    'xgb':{'base': '{{}}_xgb_seed{}', 'n':6},
    'app': {'base': '{{}}_app_seed{}', 'n':10},
    'app_prev': {'base': '{{}}_app_prev_seed{}', 'n':10},
    'app_ins': {'base': '{{}}_app_ins_seed{}', 'n':10},
    'app_credit': {'base': '{{}}_app_credit_seed{}', 'n':10},
    'app_pos': {'base': '{{}}_app_pos_seed{}', 'n':10},
    'lgb_p3': ['{}_lgb46_p3','{}_lgb46_p4','{}_lgb46_p5','{}_lgb46_p6','{}_lgb46_p7'],
    'nn': {'base': '{{}}_nn_normal_v0_seed{}', 'n':10},
    'nn2': {'base': '{{}}_nn_normal_v0_seed{}_lr0003', 'n': 7},
    #'nn': ['{}_nn_normal_v0_seed0','{}_nn_normal_v0_seed1','{}_nn_normal_v0_seed2','{}_nn_normal_v0_seed3','{}_nn_normal_v0_seed4',
    #       '{}_nn_normal_v0_seed5','{}_nn_normal_v0_seed6','{}_nn_normal_v0_seed7','{}_nn_normal_v0_seed8','{}_nn_normal_v0_seed9',
    #       '{}_nn_normal_v0_seed0_lr0003','{}_nn_normal_v0_seed1_lr0003','{}_nn_normal_v0_seed2_lr0003','{}_nn_normal_v0_seed3_lr0003']

    #'nn0',
    #'nn0_seed2',
    #'nn2_seed2'
}

base_path = '../stack/'


if __name__ == '__main__':

    df = pd.read_feather('../input/application_all.f')
    y_train = df[~df.TARGET.isnull() & (df['CODE_GENDER'] != 'XNA')].TARGET

    def load_1st_model(name, basepath='', y_train=None):
        train_oof = np.load(basepath+name.format('train')+'.npy')
        test = np.load(basepath+name.format('test')+'.npy')

        if y_train is not None:
            print('len: {}'.format(len(train_oof)))
            print("{}: {}".format(name, roc_auc_score(y_train, train_oof)))

        return train_oof, test

    def parsenames(names):
        if isinstance(names, list):
            return names
        else:
            n = int(names['n'])
            return [names['base'].format(i) for i in range(n)]

    def load_1st_model_rank_averaging(names, basepath='', y_train=None):
        trains = []
        tests = []
        for n in parsenames(names):
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
        'C': (0.002,0.0025)
    }

    print('start cv...')

    cv = GridSearchCV(LogisticRegression(penalty='l1'), parameters, verbose=3, scoring='roc_auc', cv=5, n_jobs=8)
    cv.fit(x_train, y_train)

    print('best: {}, params: {}'.format(cv.best_score_, cv.best_params_))

    cv.best_estimator_.fit(x_train, y_train)

    for i, m in enumerate(base_models):
        print('{}: {}'.format(m, cv.best_estimator_.coef_[0][i]))

    test = pd.DataFrame()

    preds = cv.best_estimator_.predict_proba(x_test)[:,0]
    test['TARGET'] = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))

    test['SK_ID_CURR'] = df[df.TARGET.isnull()].reset_index()['SK_ID_CURR']

    print(test.head(20))
    print(test.describe())
    print(test.shape)

    test.to_csv('test_stack_180829.csv',index=False)
