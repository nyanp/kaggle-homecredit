import pandas as pd
import numpy as np
import features_common


class Credit(object):
    def __init__(self, file=None):
        if file is None:
            self.credit = pd.read_feather('../input/credit_card_balance.f')
            self.transformed = False
        else:
            self.credit = pd.read_feather(file)
            self.transformed = True

    @classmethod
    def from_cache(cls):
        print('bureau loading from cache...')
        return cls('cache/credit.f')

    def fill(self):
        if self.transformed:
            return
        self.credit['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan, inplace=True)

    def transform(self):
        if self.transformed:
            return
        # 利用額 / 限度額
        # ゼロを含むとノイズが増える。
        self.credit['AMT_BALANCE_PER_LMT'] = self.credit['AMT_BALANCE'].replace(0, np.nan) / self.credit[
            'AMT_CREDIT_LIMIT_ACTUAL']
        self.credit.to_feather('cache/credit.f')
        self.transformed = True

    def _aggregate_by_prev(self, df):
        # 1回のリボルビングローンの間で、クレジット限度額が何度も変更されていることがある。
        # TODO: あってる？

        c_sorted = self.credit.sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'], ascending=False)

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
        c_prev = pd.merge(c_prev, self.credit[['SK_ID_PREV', 'SK_ID_CURR']].drop_duplicates(), on='SK_ID_PREV',
                          how='left')

        c_prev['CREDIT_LIMIT_LAST_BY_MIN'] = c_prev['CREDIT_LIMIT_LAST'] / c_prev['CREDIT_LIMIT_MIN']

        agg = features_common.group_by_1(c_prev, 'SK_ID_CURR', 'CREDIT_LIMIT_LAST_BY_MIN', 'mean',
                                         'credit_prev_mean(CREDIT_LIMIT_LAST_BY_MIN)', merge=False)

        return pd.merge(df, agg, on='SK_ID_CURR', how='left')

    def aggregate(self, df):
        print('aggregate: credit')
        num_aggregations = {
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
            'AMT_CREDIT_LIMIT_ACTUAL': ['mean']
        }

        credit_agg = self.credit.groupby('SK_ID_CURR').agg({**num_aggregations})
        credit_agg.columns = features_common.make_agg_names('credit_', credit_agg.columns.tolist())
        credit_agg.reset_index(inplace=True)
        credit_agg['CC_COUNT'] = self.credit.groupby('SK_ID_CURR').size()

        df = self._aggregate_by_prev(df)

        # 集計期間を絞る
        # TODO: この集計期間で他のCreditも試す
        c_agg12 = self.credit[self.credit['MONTHS_BALANCE'] >= -12].groupby('SK_ID_CURR').agg({
            'AMT_BALANCE_PER_LMT': ['max','mean'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['mean']
        })
        c_agg12.columns = features_common.make_agg_names('credit12_', c_agg12.columns.tolist())
        c_agg12.reset_index(inplace=True)
        df = pd.merge(df, c_agg12, on='SK_ID_CURR', how='left')

        c_agg6 = self.credit[self.credit['MONTHS_BALANCE'] >= -6].groupby('SK_ID_CURR').agg({
            'AMT_BALANCE_PER_LMT': ['min', 'max'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['mean']
        })
        c_agg6.columns = features_common.make_agg_names('credit6_', c_agg6.columns.tolist())
        c_agg6.reset_index(inplace=True)
        df = pd.merge(df, c_agg6, on='SK_ID_CURR', how='left')

        return pd.merge(df, credit_agg, on='SK_ID_CURR', how='left')
