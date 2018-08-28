import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import gc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from contextlib import contextmanager
from lightgbm import LGBMClassifier

import sys

DELAY_THRESHOLD = 14
USE_YEARS = int(sys.argv[1])

print('use years: {}'.format(USE_YEARS))

np.random.seed(42)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :50].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(9, 11))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def lgbm_cv(param, X, y, X_test, nfolds=5, submission='../output/sub.csv'):
    folds = KFold(n_splits=nfolds, shuffle=True, random_state=47)
    feature_importance_df = pd.DataFrame()
    feats = [f for f in X.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    oof_preds = np.zeros(X.shape[0])

    if X_test is not None:
        preds_test = np.empty((nfolds, X_test.shape[0]))

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X[feats], y)):
        train_x, train_y = X[feats].iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = X[feats].iloc[valid_idx], y.iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(**param)
        clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                eval_metric='auc', verbose=1000, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        if X_test is not None:
            preds_test[n_fold, :] = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:, 1]

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold {} AUC : {:.6f}'.format(n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    auc = roc_auc_score(y, oof_preds)
    print('Full AUC score %.6f' % auc)
    #display_importances(feature_importance_df)

    if submission is not None:
        preds = preds_test.mean(axis=0)
        df = pd.DataFrame()
        df['f'] = preds
        df.to_feather(submission)

    return feature_importance_df, auc


def lgbm_cv_idfold(param, X, y, X_test, nfolds=5, submission='../output/sub.csv'):
    train_ids = X.index.unique()
    len(train_ids)

    folds = KFold(n_splits=nfolds, shuffle=True, random_state=47)
    feature_importance_df = pd.DataFrame()
    feats = [f for f in X.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    oof_preds = np.zeros(X.shape[0])

    if X_test is not None:
        preds_test = np.empty((nfolds, X_test.shape[0]))

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_ids)):
        train_x, train_y = X[feats].iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = X[feats].iloc[valid_idx], y.iloc[valid_idx]

        train_x = X[X.index.isin(train_ids[train_idx])]
        train_y = y[X.index.isin(train_ids[train_idx])]

        valid_x = X[X.index.isin(train_ids[valid_idx])]
        valid_y = y[X.index.isin(train_ids[valid_idx])]

        print('train: {}, valid: {}'.format(train_x.shape, valid_x.shape))

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(**param)
        clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                eval_metric='auc', verbose=1000, early_stopping_rounds=200)

        oof_idx = np.where(X.index.isin(train_ids[valid_idx]))

        oof_preds[oof_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        if X_test is not None:
            preds_test[n_fold, :] = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:, 1]

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold {} AUC : {:.6f}'.format(n_fold + 1, roc_auc_score(valid_y, oof_preds[oof_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))
    display_importances(feature_importance_df)

    if submission is not None:
        preds = preds_test.mean(axis=0)
        df = pd.DataFrame()
        df['f'] = preds
        df.to_feather(submission)

    return feature_importance_df


with timer('load dataframes'):
    # base
    df = pd.read_feather('../input/application_all.f')

    # 信用機関からの信用情報。ユーザーごとに、過去の借入に関する情報が記録されている
    bureau = pd.read_feather('../input/bureau.f')

    bureau_balance = pd.read_feather('../input/bureau_balance.f')

    pos = pd.read_feather('../input/POS_CASH_balance.f')

    credit = pd.read_feather('../input/credit_card_balance.f')

    # 過去のローン申し込み情報
    prev = pd.read_feather('../input/previous_application.f')

    # 過去の分割払い(install)に関する支払情報
    install = pd.read_feather('../input/installments_payments.f')

if USE_YEARS > 0:
    print('before drop: {}'.format(prev.shape))
    prev = prev[prev.DAYS_DECISION >= -365*USE_YEARS]
    prev.reset_index(inplace=True, drop=True)
    print('after drop: {}'.format(prev.shape))

repl_columns = ['AMT_ANNUITY','AMT_GOODS_PRICE','NAME_TYPE_SUITE','AMT_CREDIT','NAME_CONTRACT_TYPE']
prev_train = pd.merge(prev[repl_columns+['SK_ID_CURR','SK_ID_PREV']], df.drop(repl_columns, axis=1), on='SK_ID_CURR', how='left')
prev_train.shape

pos.sort_values(by=['SK_ID_PREV','MONTHS_BALANCE'], ascending=False, inplace=True)
pos.head()

pos['delay'] = pos['SK_DPD'] > DELAY_THRESHOLD
pos.delay.value_counts()

g = pos.groupby('SK_ID_PREV')['delay'].mean().reset_index()

print(prev_train.shape)
prev_train = pd.merge(prev_train, g, on='SK_ID_PREV', how='left')
prev_train = prev_train[~prev_train['delay'].isnull()]
print(prev_train.shape)

def basic_app_features(df):
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['SOURCES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['SOURCES_STD'] = df['SOURCES_STD'].fillna(df['SOURCES_STD'].mean())
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    return df.drop('ORGANIZATION_TYPE',axis=1)

prev_train = basic_app_features(prev_train)
prev_train.shape

from lightgbm import LGBMRegressor

X = prev_train.drop(['delay','SK_ID_PREV','SK_ID_CURR','TARGET'], axis=1)
y = (prev_train['delay'] > 0.0).astype(np.int32)

print('shape: {}, target=1: {}, =0:{}'.format(X.shape, (y == 1.0).sum(), (y==0.0).sum()))

lgb_param = {
    'objective' : 'binary',
    'num_leaves' : 64,
    'learning_rate' : 0.04,
    'colsample_bytree' : 0.2,
    'subsample' : 0.872,
    'max_depth' : -1,
    'reg_alpha' : 0.04,
    'reg_lambda' : 0.073,
    'min_split_gain' : 0.0222415,
    'min_child_weight' : 20,
    'metric' : 'auc',
    'n_estimators' : 10000,
    'verbose':-1
}


def categorize(X):
    for c in X:
        if X[c].dtype.name == 'object':
            X[c] = X[c].astype('category')
    return X


X = categorize(X)

df = basic_app_features(df)
X_test = df[X.columns.tolist()]
X_test = categorize(X_test)

with open('log_prevmodel3_y.txt', 'a') as f:
    md = -1
    lr = 0.04
    cs = 0.2

    feature_importance_df, auc = lgbm_cv(lgb_param, X, y, X_test, nfolds=5, submission='test_pos5_drop_y{}.f'.format(USE_YEARS))
    print(auc)
