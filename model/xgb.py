import pandas as pd
import model_base
from xgboost import XGBClassifier
import sys

class XGBoost(model_base.ModelBase):
    def __init__(self, name, comment='', basepath='../feature/features_all.f',
                 param=None, n_estimators=None, seed=None, undersample = 0):
        super().__init__(name, comment)

        x = pd.read_feather(basepath).reset_index(drop=True)
        self.x = pd.get_dummies(x)
        
        if undersample > 0:
            print('shape(before undersampling) : {}, {}'.format(self.x[~self.x.TARGET.isnull()].shape, self.x[self.x.TARGET.isnull()].shape))

            xtest = self.x[self.x.TARGET.isnull()].reset_index(drop=True)
            xtrain = self.x[~self.x.TARGET.isnull()].reset_index(drop=True)

            cash = xtrain.query('NAME_CONTRACT_TYPE == "Cash loans"')
            revolving = xtrain.query('NAME_CONTRACT_TYPE == "Revolving loans"').sample(undersample)
            x = pd.concat([cash, revolving, xtest]).copy().reset_index(drop=True)

            print('shape(after undersampling) : {}, {}'.format(x[x.TARGET.isnull()].shape, x[x.TARGET.isnull()].shape))


        self.clf = None
        self.X_train = x[~x.TARGET.isnull()].reset_index(drop=True).drop('TARGET', axis=1)
        self.y_train = x[~x.TARGET.isnull()].reset_index(drop=True).TARGET
        self.X_test = x[x.TARGET.isnull()].reset_index(drop=True).drop('TARGET', axis=1)

        if param is None:
            self.param = {
                'objective': 'binary:logistic',
                'learning_rate': 0.01,
                'n_estimators': 10000,
                'max_depth': 4,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.5,
                'reg_lambda': 3,
                'scale_pos_weight': 2.5,
                'nthread': 16
            }
        else:
            self.param = param

        if n_estimators is not None:
            self.param['n_estimators'] = n_estimators

        if seed is not None:
            self.param['random_state'] = seed

        print('X:{}'.format(self.X_train.shape))

    def get_train(self):
        return self.X_train, self.y_train

    def get_test(self):
        return self.X_test

    def on_start_cv(self):
        self.log('param: {}'.format(self.param))

    def train(self, train_x, train_y, valid_x, valid_y) -> None:
        self.clf = XGBClassifier(**self.param)
        self.clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                     eval_metric='auc', verbose=True, early_stopping_rounds=200)

    def predict(self, test_x):
        return self.clf.predict_proba(test_x)[:, 1]


if __name__ == "__main__":
    argc = len(sys.argv)

    est = None if argc == 1 else int(sys.argv[1])

    for i in range(10):
        name = 'xgb_seed{}_uc'.format(i)
        m = XGBoost(name, 'xgb', n_estimators=None, seed=i, undersample=8000)
        m.cv(5, submission='../output/{}.csv'.format(name), save_oof='../stack/{}_' + name + '.npy')
