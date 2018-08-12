import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import time
import gc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from contextlib import contextmanager
from lightgbm import LGBMClassifier

np.random.seed(42)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def lgbm_cv(param, X, y, X_test, nfolds=5, submission='../output/sub.csv', baseline=None, fixed_epoch=True, seed=47):
    folds = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    feature_importance_df = pd.DataFrame()
    feats = [f for f in X.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    oof_preds = np.zeros(X.shape[0])

    if X_test is not None:
        preds_test = np.empty((nfolds, X_test.shape[0]))

    roc = []

    n_lose = 0
    balance = 0

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X[feats], y)):
        train_x, train_y = X[feats].iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = X[feats].iloc[valid_idx], y.iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(**param)

        if fixed_epoch:
            er = None
        else:
            er = 200

        clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                eval_metric='auc', verbose=-1, early_stopping_rounds=er)

        if fixed_epoch:
            ni = 0
        else:
            ni = clf.best_iteration_

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=ni)[:, 1]

        if submission is not None:
            preds_test[n_fold, :] = clf.predict_proba(X_test, num_iteration=ni)[:, 1]

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        # print('Fold {} AUC : {:.6f}'.format(n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        fold_auc = roc_auc_score(valid_y, oof_preds[valid_idx])

        if baseline is not None:
            balance = balance + fold_auc - baseline[n_fold]
            if fold_auc < baseline[n_fold]:
                n_lose = n_lose + 1

        roc.append(fold_auc)

        if baseline is not None and n_lose > nfolds/2 and balance < 0:
            return roc

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    # print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))
    # display_importances(feature_importance_df)

    if submission is not None:
        preds = preds_test.mean(axis=0)
        sub = pd.read_csv('../input/sample_submission.csv')
        sub['TARGET'] = preds
        sub.to_csv(submission, index=False)

    roc.append(roc_auc_score(y, oof_preds))
    return roc


def feature_selection_eval(param,
                           X: pd.DataFrame,
                           X_add,
                           y,
                           X_test,
                           nfolds,
                           set=2,
                           file='log_fw.txt',
                           earlystop=True,
                           fixed_epoch=True,
                           seed=47):
    n_columns = X_add.shape[1]

    n_loop = n_columns // set

    with open(file, 'a') as f:
        # baseline
        auc = lgbm_cv(param, X, y, None, nfolds=nfolds, submission=None, fixed_epoch=fixed_epoch, seed=seed)

        for a in auc:
            f.write('{},'.format(a))
        f.write('baseline\n')

        if earlystop:
            baseline = auc
        else:
            baseline = None

        for i in range(n_loop):
            X_c = X.copy()
            for n in range(set):
                idx = i * set + n
                col = X_add.columns.tolist()[idx]
                X_c[col] = X_add[col]

            add_columns = X_add.columns.tolist()[i * set:(i + 1) * set]
            print('add:{}'.format(add_columns))
            auc = lgbm_cv(param, X_c, y, None, nfolds=nfolds, submission=None, baseline=baseline, fixed_epoch=fixed_epoch, seed=seed)

            for a in auc:
                f.write('{},'.format(a))
            for a in add_columns:
                f.write('{},'.format(a))
            f.write('\n')
            f.flush()

def categorize(df):
    for c in df:
        if df[c].dtype.name == 'object':
            df[c] = df[c].astype('category')
    return df

import sys
argc = len(sys.argv)

add_file = 'x_add0807.f' if argc == 1 else sys.argv[1]
filename = 'log.txt' if argc < 3 else sys.argv[2]
nfolds = 5 if argc < 4 else int(sys.argv[3])
fixed = True if argc < 5 else (int(sys.argv[4]) > 0)
seed = 47 if argc < 6 else int(sys.argv[5])

print('input: {}, log: {}, folds: {}, fixed: {}, seed: {}'.format(add_file, filename, nfolds, fixed, seed))

df = pd.read_feather('../feature/features_all.f')
df = df[~df.TARGET.isnull()].reset_index(drop=True)
df = categorize(df)

X = df.drop('TARGET', axis=1)
y = pd.DataFrame()
y['y'] = df['TARGET'].astype(np.int32)

X_add = pd.read_feather(add_file)
X_add = categorize(X_add)

print(X.shape)
print(X_add.shape)
print(y.shape)

if len(X_add) > len(X) and 'SK_ID_CURR' in X_add:
    orig = [c for c in X if not c == 'SK_ID_CURR']

    X_add = pd.merge(X, X_add, on='SK_ID_CURR', how='left')
    X_add.drop(orig, axis=1, inplace=True)
    print(X_add.shape)

if 'SK_ID_CURR' in df.columns:
    df.drop('SK_ID_CURR', axis=1, inplace=True)


# lr=0.02 : 0.7888116, round1000, 976s
# lr=0.04 : 0.7888695+ 0.0022, round621, 621s
lgb_param = {
    'objective': 'binary',
    'learning_rate': 0.03,
    'max_bin':200,
    'max_depth': -1,
    'num_leaves': 30,
    'min_child_samples': 70,
    'subsample': 1.0,
    'subsample_freq': 1,
    'colsample_bytree': 0.05,
    'min_split_gain': 0.5,
    'reg_alpha': 0.0,
    'reg_lambda': 100,
    'scale_pos_weight': 1,
    'is_unbalance': False,
    'metric': 'auc',
    'n_estimators': 10000,
    'boosting_type': 'gbdt'
}

feature_selection_eval(lgb_param, X, X_add.drop('SK_ID_CURR', axis=1), y['y'], None, nfolds, set=1, file=filename,
                       fixed_epoch=True, seed=seed)
