import gc
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier


# previous_applicationからの予測値をapplicationに特徴として加える
class PrevModel(object):
    def __init__(self, name, X=7, Y=5, cutoff=0.0, param=None, seed=None):
        droplist = [ 'ORGANIZATION_TYPE',
                         'FLAG_DOCUMENT_5',
                         'FLAG_DOCUMENT_6',
                         'FLAG_DOCUMENT_8',
                         'FLAG_DOCUMENT_16',
                         'FLAG_DOCUMENT_18',
                         'FLAG_CONT_MOBILE',
                         'FLAG_DOCUMENT_10',
                         'FLAG_DOCUMENT_11',
                         'FLAG_DOCUMENT_12',
                         'FLAG_DOCUMENT_13',
                         'FLAG_DOCUMENT_14',
                         'FLAG_DOCUMENT_15',
                         'FLAG_DOCUMENT_17',
                         'FLAG_DOCUMENT_19',
                         'FLAG_DOCUMENT_2',
                         'FLAG_DOCUMENT_20',
                         'FLAG_DOCUMENT_21',
                         'FLAG_DOCUMENT_4',
                         'FLAG_DOCUMENT_7',
                         'FLAG_DOCUMENT_9',
                         'FLAG_MOBIL'
                        ]

        self.name = name
        self.x_train, self.x_test = self._load_base()
        self.x_train = self._make_target_variable(self.x_train, X, Y, cutoff)
        self.x_train = self.basic_features(self.x_train)
        self.x_test = self.basic_features(self.x_test)

        self.x_train, self.x_test = self.custom_features(self.x_train, self.x_test)
        self.x_train = self.categorize(self.x_train)
        self.x_test = self.categorize(self.x_test)

        self.x_train.drop(droplist, axis=1, inplace=True)
        self.x_test.drop(droplist, axis=1, inplace=True)

        self.y_train = self.x_train['delay']
        self.y_target = self.x_test['TARGET']

        self.x_train.drop(['delay', 'TARGET'], axis=1, inplace=True)
        self.x_test.drop(['TARGET'], axis=1, inplace=True)

        self.classifiers = []

        assert self.x_train.shape[1] == self.x_test.shape[1]

        self.logfile = open('../output/{}.txt'.format(name), 'w')

        if param is None:
            self.param = {
                'objective': 'binary',
                'num_leaves': 32,
                'learning_rate': 0.04,
                'colsample_bytree': 0.95,
                'subsample': 0.872,
                'max_depth': 8,
                'reg_alpha': 0.04,
                'reg_lambda': 0.073,
                'min_split_gain': 0.0222415,
                'min_child_weight': 40,
                'metric': 'auc',
                'n_estimators': 10000
            }
        else:
            self.param = param

        if seed is not None:
            self.param['seed'] = seed


    def custom_features(self, x_train, x_test):
        return x_train, x_test


    def categorize(self, X):
        for c in X:
            if X[c].dtype.name == 'object':
                X[c] = X[c].astype('category')
        return X

    def basic_features(self, df):
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
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

        if 'SK_ID_CURR' in df:
            df.drop('SK_ID_CURR', axis=1, inplace=True)

        if 'SK_ID_PREV' in df:
            df.drop('SK_ID_PREV', axis=1, inplace=True)

        return df

    def _load_base(self):
        # merge id information to previous application
        prev = pd.read_feather('../input/previous_application.f')
        df = pd.read_feather('../input/application_all.f')

        repl_columns = ['AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'AMT_CREDIT', 'NAME_CONTRACT_TYPE']
        prev_train = pd.merge(prev[repl_columns + ['SK_ID_CURR', 'SK_ID_PREV']], df.drop(repl_columns, axis=1),
                              on='SK_ID_CURR', how='left')
        print(prev_train.shape)
        print(df.shape)
        return prev_train, df

    # make target variable
    def _make_target_variable(self, prev_train, X=7, Y=5, cutoff=0.0):
        # 1 - client with payment difficulties: he/she had late payment more than *X* days
        #     on at least one of the first *Y* installments of the loan in our sample,
        # 0 - all other cases
        install = pd.read_feather('../input/installments_payments.f')
        install['DPD'] = install['DAYS_ENTRY_PAYMENT'] - install['DAYS_INSTALMENT']
        install['DPD'] = install['DPD'].apply(lambda x: x if x > 0 else 0)

        if Y > 0:
            install_head = install[install.NUM_INSTALMENT_NUMBER <= Y]
        else:
            install_head = install  # [install.NUM_INSTALMENT_NUMBER <= 5]

        install_head['delay'] = (install_head['DPD'] > X).astype(np.int32)

        g = install_head.groupby('SK_ID_PREV')['delay'].mean().reset_index()

        prev_train = pd.merge(prev_train, g, on='SK_ID_PREV', how='left')
        prev_train = prev_train[~prev_train['delay'].isnull()]

        prev_train['delay'] = (prev_train['delay'] > cutoff).astype(np.int32)

        print(prev_train.delay.value_counts())

        return prev_train


    def cv(self, nfolds=5, submission=True, feature_name='PREDICTED_DPD'):
        self.classifiers.clear()

        folds = KFold(n_splits=nfolds, shuffle=True, random_state=47)
        self.feature_importance_df = pd.DataFrame()

        oof_preds = np.zeros(self.x_train.shape[0])
        preds_test = np.empty((nfolds, self.x_test.shape[0]))

        self.logfile.write('param: {}\n'.format(self.param))
        self.logfile.write('fold: {}\n'.format(nfolds))
        self.logfile.write('data shape: {}\n'.format(self.x_train.shape))
        self.logfile.write('features: {}\n'.format(self.x_train.columns.tolist()))
        self.logfile.write('output: ../output/{}.csv\n'.format(self.name))
        self.logfile.flush()

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(self.x_train, self.y_train)):
            fstart = time.time()
            train_x, train_y = self.x_train.iloc[train_idx], self.y_train.iloc[train_idx]
            valid_x, valid_y = self.x_train.iloc[valid_idx], self.y_train.iloc[valid_idx]

            # LightGBM parameters found by Bayesian optimization
            clf = LGBMClassifier(**self.param)
            clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                    eval_metric='auc', verbose=25, early_stopping_rounds=200)

            oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
            preds_test[n_fold, :] = clf.predict_proba(self.x_test, num_iteration=clf.best_iteration_)[:, 1]

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = self.x_train.columns.tolist()
            fold_importance_df["importance"] = clf.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], axis=0)

            strlog = '[{}][{:.1f} sec] Fold {} AUC : {:.6f}'.format(str(datetime.now()), time.time() - fstart, n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx]))
            print(strlog)
            self.logfile.write(strlog+'\n')

            self.classifiers.append(clf)
            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()

        strlog = 'Full AUC score {:.6f}'.format(roc_auc_score(self.y_train, oof_preds))
        print(strlog)
        self.logfile.write(strlog+'\n')
        #display_importances(self.feature_importance_df)

        if submission:
            preds = preds_test.mean(axis=0)
            sub = pd.DataFrame()
            sub[feature_name] = preds
            sub.to_csv('../output/preds_{}.csv'.format(self.name), index=False)

            sub_corr = pd.DataFrame()
            sub_corr['predicted'] = preds
            sub_corr['actual'] = self.y_target
            print(sub_corr.corr())
            self.logfile.write(sub_corr.corr().to_string())

        return self.feature_importance_df



if __name__ == "__main__":
    for seed in range(10):
        m = PrevModel(name='prevmodel_x14y-1_seed{}'.format(seed), X=14, Y=-1)
        m.cv()
