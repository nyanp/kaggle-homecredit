import pandas as pd
import credit
import bureau
import application
import pos_cash
import install
import prev
import time
import os
import sys
import zero_importance
import features_common
import add0812

class Feature(object):
    def __init__(self, update='all', prep_mode=0):
        tables = {'credit': credit.Credit() if update == 'all' or ('credit' in update) else credit.Credit.from_cache(),
                  'bureau': bureau.Bureau(prep_mode=prep_mode) if update == 'all' or ('bureau' in update) or (prep_mode > 0) else bureau.Bureau.from_cache(),
                  'prev': prev.Prev() if update == 'all' or ('prev' in update) else prev.Prev.from_cache(),
                  'install': install.Install() if update == 'all' or ('install' in update) else install.Install.from_cache(),
                  'cash': pos_cash.PosCash() if update == 'all' or ('cash' in update) else pos_cash.PosCash.from_cache(),
                  'app': application.Application() if update == 'all' or ('app' in update) else application.Application.from_cache()}

        print('transform...')
        for k, v in tables.items():
            print(k)
            v.fill()
            v.transform()

        df = tables['app'].df

        for k, v in tables.items():
            if k == 'app':
                continue

            df = v.aggregate(df)
            print('{} merged. shape: {}'.format(k, df.shape))

        df = tables['app'].transform_with_others(df, tables['prev'].df, tables['cash'].df, tables['credit'].df)
        print('transform finished. {}'.format(df.shape))

        self.tables = tables
        self.df = df
        self._load_exotic()
        self._load_add()
        self._delete_columns()

    def _delete_columns(self):
        self.df.drop(['ORGANIZATION_TYPE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                      'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
                      'FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
                      'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_17',
                      'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_4',
                      'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_MOBIL'
                      ], axis=1, inplace=True)

        drops = [c for c in zero_importance.useless_features if c in self.df]
        print('drop useless columns: {}'.format(drops))

        self.df.drop(drops, axis=1, inplace=True)

    def _load_exotic(self):
        d = pd.read_feather('../model/predicted_dpd.ftr')
        self.df = pd.merge(self.df, d[['SK_ID_CURR', 'PREDICTED_X14Y-1']], on='SK_ID_CURR', how='left')

        d = pd.read_feather('../model/x_add2.ftr')
        self.df = pd.merge(self.df, d[['SK_ID_CURR', 'POS_PREDICTED']], on='SK_ID_CURR', how='left')

        d = pd.read_feather('p.ftr')
        self.df = pd.merge(self.df, d[['SK_ID_CURR', 'PRED_DPD_P_INS']], on='SK_ID_CURR', how='left')


    def _load_add(self):
        df = features_common.read_application()
        prev = features_common.read_csv('../input/previous_application.csv')
        install = features_common.read_csv('../input/installments_payments.csv')
        bureau = features_common.read_csv('../input/bureau.csv')
        bb = features_common.read_csv('../input/bureau_balance.csv')
        pos = features_common.read_csv('../input/POS_CASH_balance.csv')
        credit = features_common.read_csv('../input/credit_card_balance.csv')

        if os.path.exists('cache/0812.f'):
            f1 = pd.read_feather('cache/0812.f')
        else:
            print('make additional features(0812)...')
            f1 = add0812.make_features(df, prev, bureau, bb, pos, credit, install)
            f1.to_feather('cache/0812.f')

        self.df = pd.merge(self.df, f1, on='SK_ID_CURR', how='left')


if __name__ == "__main__":
    if not os.path.exists('cache'):
        os.makedirs('cache')

    argc = len(sys.argv)

    if argc >= 2:
        if sys.argv[1] == 'nocache':
            update = 'all'
        elif sys.argv[1] == 'cache':
            update = []
        else:
            update = sys.argv[1]
    else:
        update = []

    save_file = 'features_all.f' if argc <= 2 else sys.argv[2]
    prep_mode = 0 if argc <= 3 else int(sys.argv[3])

    start = time.time()
    f = Feature(update=update, prep_mode=prep_mode)

    print(f.df.shape)

    # 実装に依存して順番が変わるとCVが変わってしまうので、最後にソートしておく。
    x = f.df[sorted(f.df.columns)]

    f.df.to_feather(save_file)
    f.df.head(100).to_csv('all_sample.csv', index=False)

    print('finished generating features. ({} sec)'.format(time.time() - start))
