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

        self.df.sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'], ascending=False, inplace=True)

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

    def _aggregate_by_prev(self, df_base):
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

        df_base = pd.merge(df_base, agg, on='SK_ID_CURR', how='left')

        # 現在Activeなクレジットの返済残高の合計
        prev = pd.read_feather('../input/previous_application.f')

        df_active_balance = features_common.extract_active_balance(self.df)
        df_active_loans = df_active_balance.groupby('SK_ID_PREV')\
                                           .head(1)[['SK_ID_CURR', 'SK_ID_PREV', 'AMT_BALANCE']]\
                                           .rename(columns={'AMT_BALANCE': 'AMT_CREDIT_SUM_DEBT_CREDIT'})

        df_active_loans = pd.merge(df_active_loans, prev.drop('SK_ID_CURR',axis=1), on='SK_ID_PREV', how='left')

        df_active_loans['NAME_YIELD_GROUP_high'] = (df_active_loans.NAME_YIELD_GROUP == 'high').astype(np.int32)
        df_active_loans['NAME_YIELD_GROUP_low_normal'] = (df_active_loans.NAME_YIELD_GROUP == 'low_normal').astype(np.int32)
        df_active_loans['NAME_YIELD_GROUP_low_action'] = (df_active_loans.NAME_YIELD_GROUP == 'low_action').astype(np.int32)
        df_active_loans['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        df_active_loans['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)

        agg = {
            'DAYS_LAST_DUE_1ST_VERSION': ['mean'],
            'DAYS_FIRST_DUE': ['mean'],
            'CNT_PAYMENT': ['mean'],
            'AMT_ANNUITY': ['sum'],
            'AMT_GOODS_PRICE': ['sum'],
            'NAME_YIELD_GROUP_high': ['mean'],
            'NAME_YIELD_GROUP_low_normal': ['mean'],
            'NAME_YIELD_GROUP_low_action': ['mean'],
            'AMT_CREDIT_SUM_DEBT_CREDIT': ['sum']
        }

        agg = df_active_loans.groupby('SK_ID_CURR').agg(agg)
        agg.columns = features_common.make_agg_names('credit_active_', agg)
        agg.reset_index(inplace=True)

        agg.rename(columns={'credit_active_sum(AMT_CREDIT_SUM_DEBT_CREDIT)' : 'SUM(AMT_DEBT_ACTIVE_LOAN_CREDIT)'}, inplace=True)

        return pd.merge(df_base, agg, on='SK_ID_CURR', how='left')

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
