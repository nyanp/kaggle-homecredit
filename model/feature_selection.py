import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import lightgbm as lgb
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


def lgbm_cv(param, X, y, X_test, nfolds=5, submission='../output/sub.csv'):
    folds = KFold(n_splits=nfolds, shuffle=True, random_state=47)
    feature_importance_df = pd.DataFrame()
    feats = [f for f in X.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    oof_preds = np.zeros(X.shape[0])

    if X_test is not None:
        preds_test = np.empty((nfolds, X_test.shape[0]))

    roc = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X[feats], y)):
        train_x, train_y = X[feats].iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = X[feats].iloc[valid_idx], y.iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(**param)
        clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                eval_metric='auc', verbose=-1, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        if submission is not None:
            preds_test[n_fold, :] = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:, 1]

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        #print('Fold {} AUC : {:.6f}'.format(n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        roc.append(roc_auc_score(valid_y, oof_preds[valid_idx]))

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    #print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))
    #display_importances(feature_importance_df)

    if submission is not None:
        preds = preds_test.mean(axis=0)
        sub = pd.read_csv('../input/sample_submission.csv')
        sub['TARGET'] = preds
        sub.to_csv(submission, index=False)

    roc.append(roc_auc_score(y, oof_preds))
    return roc


def feature_selection_eval(param, X:pd.DataFrame, y, X_test, nfolds, set=3, file='log.txt'):
    n_columns = X.shape[1]

    n_loop = n_columns // set

    with open(file, 'a') as f:
        # baseline
        auc = lgbm_cv(param, X, y, None, 5, None)
        f.write('{},{},{},{},{},{}'.format(auc[0], auc[1], auc[2], auc[3], auc[4], auc[5]))

        for i in range(n_loop):
            drop_columns = X.columns.tolist()[i*set:(i+1)*set]
            print('drop:{}'.format(drop_columns))
            auc = lgbm_cv(param, X.drop(drop_columns, axis=1), y, None, 5, None)
            f.write('{},{},{},{},{},{},{}\n'.format(auc[0],auc[1],auc[2],auc[3],auc[4],auc[5],drop_columns[0]))
            f.flush()

#X = pd.read_feather('x.f')
#y = pd.read_feather('y.f')
df = pd.read_feather('x_model12.f')
df = df[~df.TARGET.isnull()]
X = df.drop('TARGET',axis=1)
y = df[['TARGET']]
del df

print(X.shape)
print(y.shape)

print(y.head())

# lr=0.02 : 0.7888116, round1000, 976s
# lr=0.04 : 0.7888695+ 0.0022, round621, 621s
lgb_param = {
    'objective' : 'binary',
    'num_leaves' : 32,
    'learning_rate' : 0.04,
    'colsample_bytree' : 0.2,
    'subsample' : 0.872,
    'max_depth' : 8,
    'reg_alpha' : 0.04,
    'reg_lambda' : 0.073,
    'min_split_gain' : 0.0222415,
    'min_child_weight' : 80,
    'metric' : 'auc',
    'n_estimators' : 10000,
    'verbose': -1
}

feature_selection_eval(lgb_param, X, y['TARGET'], None, 5, 1, file='log_fs180727.txt')
