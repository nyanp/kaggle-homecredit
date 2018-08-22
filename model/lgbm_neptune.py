import pandas as pd
import model_base
from lightgbm import LGBMClassifier


class LGBM_Neptune(model_base.ModelBase):
    def __init__(self, name, comment='', dtrain='../feature/feather307511.f', dtest='../feature/feather48744.f',
                 param=None, n_estimators = None):
        super().__init__(name, comment)

        df = pd.read_feather('../input/application_all.f')

        self.clf = None
        self.X_train = pd.read_feather(dtrain)
        self.y_train = df[~df.TARGET.isnull()].reset_index().TARGET
        self.X_test = pd.read_feather(dtest)

        assert self.X_train.shape[1] == self.X_test.shape[1]
        assert self.X_train.shape[0] == self.y_train.shape[0]

        if param is None:
            self.param = {
                'objective': 'binary',
                'learning_rate': 0.02,
                'max_bin':400,
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
        self.clf = LGBMClassifier(**self.param)
        self.clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                     eval_metric='auc', verbose=25, early_stopping_rounds=200)

    def predict(self, test_x):
        return self.clf.predict_proba(test_x, num_iteration=self.clf.best_iteration_)[:, 1]


if __name__ == "__main__":

    m = LGBM_Neptune('neptune', 'dataframe from open solution', n_estimators=100)

    m.cv(5, submission='../output/neptune.csv', save_oof='../stack/{}_neptune.npy')
