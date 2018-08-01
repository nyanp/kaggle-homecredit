import pandas as pd
import numpy as np
import features_common


class PosCash(object):
    def __init__(self, file=None):
        if file is None:
            self.df = pd.read_feather('../input/POS_CASH_balance.f')
            self.df.reset_index(inplace=True, drop=True)
            self.transformed = False
        else:
            self.df = pd.read_feather(file)
            self.transformed = True

        self.df.sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'], ascending=False, inplace=True)

    @classmethod
    def from_cache(cls):
        print('cash loading from cache...')
        return cls('cache/pos.f')

    def fill(self):
        pass

    def transform(self):
        if self.transformed:
            return

        normal = ['Active','Completed','Signed','Approved']
        self.df['IRREGULAR_CONTRACT'] = self.df.NAME_CONTRACT_STATUS.isin(normal).astype(np.int32)
        print(self.df.IRREGULAR_CONTRACT.value_counts())

        self.df.to_feather('cache/pos.f')
        self.transformed = True

    def _aggregate_per_loan(self, df):
        completed_ids = df[df.NAME_CONTRACT_STATUS == 'Completed'].SK_ID_PREV.unique()

        # additional
        # CNT_INSTALMENT_AHEAD_RATIO:前倒しの割合
        # CNT_INSTALMENT_AHEAD:前倒しの月数

        # Done: DPDやDPD_DEFも、期間を切ってみる -> 効かず
        def calc_aheads(df):
            tail = df.groupby('SK_ID_PREV')['SK_ID_CURR', 'SK_ID_PREV', 'CNT_INSTALMENT'].tail(1)
            head = df.groupby('SK_ID_PREV')['SK_ID_CURR', 'SK_ID_PREV', 'CNT_INSTALMENT'].head(1)
            tail.rename(columns={'CNT_INSTALMENT': 'PLANNED_CNT_INSTALMENT'}, inplace=True)
            head.rename(columns={'CNT_INSTALMENT': 'ACTUAL_CNT_INSTALMENT'}, inplace=True)
            tail = pd.merge(tail, head[['SK_ID_PREV', 'ACTUAL_CNT_INSTALMENT']], on='SK_ID_PREV', how='left')
            tail['CNT_INSTALMENT_AHEAD'] = tail['PLANNED_CNT_INSTALMENT'] - tail['ACTUAL_CNT_INSTALMENT']
            tail['CNT_INSTALMENT_AHEAD_RATIO'] = tail['ACTUAL_CNT_INSTALMENT'] / tail['PLANNED_CNT_INSTALMENT']
            return tail

        df_agg_aheads = calc_aheads(df)
        df_agg_aheads['N_LOANS'] = 1
        df_agg_aheads['IS_ACTIVE'] = (~df.SK_ID_PREV.isin(completed_ids)).astype(np.int32)

        return df_agg_aheads

    def _aggregate_by_prev2(self, df_base):
        # アクティブなローンの返済残高
        # POS_CASHだけからは分からないので、1回あたりの返済額をprevから引っ張ってきて、残りの返済回数と掛けて算出
        prev = pd.read_feather('../input/previous_application.f')

        df_active_balance = features_common.extract_active_balance(self.df)
        df_active_loans = df_active_balance.groupby('SK_ID_PREV')[['SK_ID_CURR', 'CNT_INSTALMENT_FUTURE']].min().reset_index()

        df_active_loans = pd.merge(df_active_loans, prev.drop('SK_ID_CURR',axis=1), on='SK_ID_PREV', how='left')
        df_active_loans['AMT_CREDIT_SUM_DEBT_POS'] = df_active_loans['AMT_ANNUITY'] * df_active_loans['CNT_INSTALMENT_FUTURE']

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
            'AMT_CREDIT_SUM_DEBT_POS': ['sum']
        }

        agg = df_active_loans.groupby('SK_ID_CURR').agg(agg)
        agg.columns = features_common.make_agg_names('pos_active_', agg)
        agg.reset_index(inplace=True)

        agg.rename(columns={'pos_active_sum(AMT_CREDIT_SUM_DEBT_POS)' : 'SUM(AMT_DEBT_ACTIVE_LOAN_POS)'}, inplace=True)

        df_base = pd.merge(df_base, agg, on='SK_ID_CURR', how='left')

        # 一番annuityが大きいものを代表として、直接dfの特徴に加える
        df_active_last = df_active_loans.sort_values(by=['AMT_ANNUITY'], ascending=False)\
                                        .drop_duplicates(subset=['SK_ID_CURR'], keep='first')

        df_active_last['CREDIT_TO_ANNUITY_RATIO'] = df_active_last['AMT_CREDIT'] / df_active_last['AMT_ANNUITY']
        df_active_last['CREDIT_TO_GOODS_RATIO'] = df_active_last['AMT_GOODS_PRICE'] / df_active_last['AMT_CREDIT']

        columns = ['CNT_PAYMENT', 'CREDIT_TO_GOODS_RATIO', 'DAYS_DECISION', 'AMT_ANNUITY', 'NAME_TYPE_SUITE',
                   'PRODUCT_COMBINATION', 'NAME_YIELD_GROUP']
        df_active_last = df_active_last[['SK_ID_CURR'] + columns]

        columns_new = ['p' + c for c in columns]
        df_active_last.columns = ['SK_ID_CURR'] + columns_new
        df_base = pd.merge(df_base, df_active_last, on='SK_ID_CURR', how='left')

        return df_base


    def aggregate(self, df_base):
        print('aggregate: cash')

        num_aggregations = {
            'MONTHS_BALANCE': ['count', 'mean'],
            'SK_DPD': ['sum', 'max'],
            'SK_DPD_DEF': ['sum', 'max'],
            'IRREGULAR_CONTRACT': ['sum']
        }

        df_base = features_common.aggregate(df_base, num_aggregations, self.df, 'pos_')
        df_base = features_common.aggregate(df_base, num_aggregations, self.df.query('MONTHS_BALANCE >= -12'), 'pos12_')

        pos_prev = self._aggregate_per_loan(self.df)
        pos_prev12 = self._aggregate_per_loan(self.df[self.df.MONTHS_BALANCE >= -12])
        pos_prev12.head()

        pos_agg = pos_prev.groupby('SK_ID_CURR').agg({
            'CNT_INSTALMENT_AHEAD': ['min', 'max'],
            'CNT_INSTALMENT_AHEAD_RATIO': ['min'],
            #'IS_ACTIVE': ['count']
        })

        pos_agg.columns = features_common.make_agg_names('pos_', pos_agg.columns.tolist())
        pos_agg.reset_index(inplace=True)

        pos12_agg = pos_prev12.groupby('SK_ID_CURR').agg({
            'CNT_INSTALMENT_AHEAD': ['min', 'max'],
            'CNT_INSTALMENT_AHEAD_RATIO': ['mean', 'max']
        })

        pos12_agg.columns = features_common.make_agg_names('pos12_', pos12_agg.columns.tolist())
        pos12_agg.reset_index(inplace=True)

        pos_all = pd.merge(pos_agg, pos12_agg, on='SK_ID_CURR', how='left')

        df_base = pd.merge(df_base, pos_all, on='SK_ID_CURR', how='left')

        df_base = self._aggregate_by_prev2(df_base)

        return df_base
