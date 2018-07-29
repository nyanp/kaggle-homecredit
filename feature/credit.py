import pandas as pd
import numpy as np
import features_common
import numpy as np

class Credit(object):
    def __init__(self, file=None):
        if file is None:
            self.df = pd.read_feather('../input/credit_card_balance.f')
            self.transformed = False
        else:
            self.df = pd.read_feather(file)
            self.transformed = True

    @classmethod
    def from_cache(cls):
        print('credit loading from cache...')
        return cls('cache/credit.f')

    def fill(self):
        if self.transformed:
            return
        self.df['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan, inplace=True)

    def transform(self):
        if self.transformed:
            return
        # 利用額 / 限度額
        # ゼロを含むとノイズが増える。
        self.df['AMT_BALANCE_PER_LMT'] = self.df['AMT_BALANCE'].replace(0, np.nan) / self.df[
            'AMT_CREDIT_LIMIT_ACTUAL']

        self.df.to_feather('cache/credit.f')
        self.transformed = True

    def _aggregate_by_prev(self, df):
        # 1回のリボルビングローンの間で、クレジット限度額が何度も変更されていることがある。
        # TODO: あってる？

        c_sorted = self.df.sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'], ascending=False)

        c_prev = features_common.group_by_1(c_sorted,
                                            'SK_ID_PREV',
                                            'AMT_CREDIT_LIMIT_ACTUAL',
                                            'min',
                                            'CREDIT_LIMIT_MIN',
                                            merge=False)

        c_last = c_sorted[~c_sorted['AMT_CREDIT_LIMIT_ACTUAL'].isnull()].groupby('SK_ID_PREV')[
            'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL'].head(1).reset_index(drop=True)
        c_last.columns = ['SK_ID_PREV', 'CREDIT_LIMIT_LAST']

        c_prev = pd.merge(c_prev, c_last, on='SK_ID_PREV', how='left')
        c_prev = pd.merge(c_prev, self.df[['SK_ID_PREV', 'SK_ID_CURR']].drop_duplicates(), on='SK_ID_PREV',
                          how='left')

        c_prev['CREDIT_LIMIT_LAST_BY_MIN'] = c_prev['CREDIT_LIMIT_LAST'] / c_prev['CREDIT_LIMIT_MIN']

        agg = features_common.group_by_1(c_prev, 'SK_ID_CURR', 'CREDIT_LIMIT_LAST_BY_MIN', 'mean',
                                         'credit_prev_mean(CREDIT_LIMIT_LAST_BY_MIN)', merge=False)

        return pd.merge(df, agg, on='SK_ID_CURR', how='left')

    def aggregate(self, df_base):
        print('aggregate: credit')
        agg = {
            'MONTHS_BALANCE': ['count', 'mean'],
            'SK_DPD': ['sum', 'max'],
            'SK_DPD_DEF': ['sum', 'max'],

            'AMT_INST_MIN_REGULARITY': ['mean'],
            'AMT_PAYMENT_CURRENT': ['mean'],

            'CNT_DRAWINGS_ATM_CURRENT': ['mean'],
            'CNT_DRAWINGS_CURRENT': ['mean'],
            # 'CNT_DRAWINGS_OTHER_CURRENT': ['mean'],
            'CNT_DRAWINGS_POS_CURRENT': ['mean'],
            'CNT_INSTALMENT_MATURE_CUM': ['mean'],

            'AMT_DRAWINGS_CURRENT': ['mean'],
            'AMT_DRAWINGS_OTHER_CURRENT': ['mean'],
            'AMT_DRAWINGS_POS_CURRENT': ['mean'],
            'AMT_DRAWINGS_ATM_CURRENT': ['mean'],

            'AMT_BALANCE_PER_LMT': ['max', 'mean'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['mean'],
        }
        agg12 = {
            'AMT_BALANCE_PER_LMT': ['max', 'mean'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['mean']
        }
        agg6 = {
            'AMT_BALANCE_PER_LMT': ['min', 'max'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['mean']
        }

        df_base = self._aggregate_by_prev(df_base)

        df_base = features_common.aggregate(df_base, agg, self.df, 'credit_', count_column='CC_COUNT')
        df_base = features_common.aggregate(df_base, agg12, self.df.query('MONTHS_BALANCE >= -12'), 'credit12_', count_column=None)
        df_base = features_common.aggregate(df_base, agg6, self.df.query('MONTHS_BALANCE >= -6'), 'credit6_', count_column=None)

        return df_base
