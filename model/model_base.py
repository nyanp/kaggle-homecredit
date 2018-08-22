import pandas as pd
import gc
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


class ModelBase(object):
    def __init__(self, name, comment):
        self.logfile = open('../output/{}.txt'.format(name), 'w')
        self.log('commnet: {}'.format(comment))

    def log(self, msg: str):
        self.logfile.write(msg+'\n')

    def train(self, train_x, train_y, valid_x, valid_y) -> None:
        raise NotImplementedError()

    def predict(self, test_x):
        raise NotImplementedError()

    def get_train(self):
        raise NotImplementedError()

    def get_test(self):
        raise NotImplementedError()

    def on_start_cv(self):
        pass

    def cv(self, nfolds=5, submission=None, seed=47, save_oof=None):
        x_train, y_train = self.get_train()
        x_test = self.get_test()

        folds = KFold(n_splits=nfolds, shuffle=True, random_state=seed)

        self.log('n-folds: {}'.format(nfolds))
        self.log('seed: {}'.format(seed))
        self.log('sub: {}'.format(submission))
        self.log('oof: {}'.format(save_oof))
        self.on_start_cv()

        oof_preds = np.zeros(x_train.shape[0])
        preds_test = np.empty((nfolds, x_test.shape[0]))

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x_train, y_train)):
            fstart = time.time()
            train_x, train_y = x_train.iloc[train_idx], y_train.iloc[train_idx]
            valid_x, valid_y = x_train.iloc[valid_idx], y_train.iloc[valid_idx]

            # LightGBM parameters found by Bayesian optimization

            self.train(train_x, train_y, valid_x, valid_y)

            oof_preds[valid_idx] = self.predict(valid_x)
            preds_test[n_fold, :] = self.predict(x_test)

            strlog = '[{}][{:.1f} sec] Fold {} AUC : {:.6f}'.format(str(datetime.now()), time.time() - fstart, n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx]))
            print(strlog)
            self.logfile.write(strlog+'\n')
            self.logfile.flush()

            del train_x, train_y, valid_x, valid_y
            gc.collect()

        full_auc = roc_auc_score(y_train, oof_preds)
        strlog = 'Full AUC score {:.6f}'.format(full_auc)
        print(strlog)
        self.logfile.write(strlog + '\n')
        self.logfile.flush()

        preds = preds_test.mean(axis=0)

        if submission is not None:
            sub = pd.read_csv('../input/sample_submission.csv')
            sub['TARGET'] = preds
            sub.to_csv('../output/{}.csv'.format(submission), index=False)

        if save_oof is not None:
            np.save(save_oof.format('train'), oof_preds)
            np.save(save_oof.format('test'), preds)

        return full_auc, oof_preds, preds
