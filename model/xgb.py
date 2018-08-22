import pandas as pd
import model_base
from xgboost import XGBClassifier
import sys

class XGBoost(model_base.ModelBase):
    def __init__(self, name, comment='', basepath='../feature/features_all.f',
                 param=None, n_estimators=None):
        super().__init__(name, comment)

        x = pd.read_feather(basepath).reset_index(drop=True)

        self.clf = None
        self.X_train = x[~x.TARGET.isnull()].reset_index(drop=True).drop('TARGET', axis=1)
        self.y_train = x[~x.TARGET.isnull()].reset_index(drop=True).TARGET
        self.X_test = x[x.TARGET.isnull()].reset_index(drop=True)

        if param is None:
            self.param = {
                'objective': 'binary:logistic',
                'learning_rate': 0.01,
                'n_estimators': 10000,
                'max_depth': 4,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_lambda': 1.2,
                'scale_pos_weight': 2.5
            }
        else:
            self.param = param

        if n_estimators is not None:
            self.param['n_estimators'] = n_estimators

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
                     eval_metric='auc', verbose=25, early_stopping_rounds=200)

    def predict(self, test_x):
        return self.clf.predict_proba(test_x)[:, 1]


if __name__ == "__main__":
    argc = len(sys.argv)

    est = None if argc == 1 else int(sys.argv[1])

    m = XGBoost('xgb', 'xgb', n_estimators=est)

    m.cv(5, submission='../output/xgb.csv', save_oof='../stack/{}_xgb.npy')
