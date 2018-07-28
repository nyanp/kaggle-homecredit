import pandas as pd
import numpy as np
import features_common


class Bureau(object):
    def __init__(self):
        self.bureau = pd.read_feather('../input/bureau.f')
        self.bb = pd.read_feather('../input/bureau_balance.f')

    def fill(self):
        #self.bureau['AMT_CREDIT_SUM'].fillna(0, inplace=True)
        self.bureau['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)

    def transform(self):
        bureau = self.bureau
        bureau_balance = self.bb
        
        # 終了予定日と、実際の終了日の差
        bureau['ENDDATE_DIFF'] = bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT_ENDDATE']

        # 返済期間の長さ
        bureau['DAYS_CREDIT_PLAN'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT']

        # 返済遅延
        bureau_balance['DPD'] = bureau_balance[['STATUS']].replace(['C', 'X', '0', '1', '2', '3', '4', '5'],
                                                                             [0, 0, 0, 15, 45, 75, 105, 135])
        bureau_balance['DPD'].value_counts()

        # 同一のCREDIT_TYPEの中での、相対クレジット額
        mean_amt_credit = bureau.groupby('CREDIT_TYPE')['AMT_CREDIT_SUM'].quantile().reset_index()
        mean_amt_credit.columns = ['CREDIT_TYPE', 'MEAN_AMT_CREDIT_BY_CREDIT_TYPE']
        bureau = pd.merge(bureau, mean_amt_credit, on='CREDIT_TYPE', how='left')
        bureau['AMT_CREDIT_SUM_RATIO'] = bureau['AMT_CREDIT_SUM'] / bureau['MEAN_AMT_CREDIT_BY_CREDIT_TYPE']
    
        # 限度額に対する借入額の比率
        bureau['AMT_CREDIT_DEBT_PERC'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM']
        bureau['AMT_CREDIT_DEBT_DIFF'] = bureau['AMT_CREDIT_SUM_DEBT'] - bureau['AMT_CREDIT_SUM']
    
        # 総額と1回あたり支払額の比率
        bureau['AMT_CREDIT_ANNUITY_PERC'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']

        # TODO: こっちだけで良さそうなら前処理へ。
        b_replaced = bureau.copy()
        b_replaced['AMT_CREDIT_SUM_DEBT'].replace(0, np.nan, inplace=True)
        bureau['AMT_CREDIT_DEBT_PERC_NZ'] = b_replaced['AMT_CREDIT_SUM_DEBT'] / b_replaced['AMT_CREDIT_SUM']
        bureau['AMT_CREDIT_DEBT_DIFF_NZ'] = b_replaced['AMT_CREDIT_SUM'] - b_replaced['AMT_CREDIT_SUM_DEBT']

        self.bureau = bureau
        self.bb = bureau_balance

    def aggregate(self, df):
        print('aggregate: bureau')
        agg = {
            'DAYS_CREDIT': ['count', 'mean'],
            'CREDIT_DAY_OVERDUE': ['mean', 'sum'],
            'DAYS_CREDIT_ENDDATE': ['mean'],
            'DAYS_ENDDATE_FACT': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean', 'sum'],
            'CNT_CREDIT_PROLONG': ['mean', 'sum'],
            'AMT_CREDIT_SUM': ['mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean', 'sum'],
            'CREDIT_TYPE': ['nunique'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'AMT_ANNUITY': ['mean', 'sum'],

            'BB_DPD_MEAN': ['mean', 'max'],
            'BB_MONTHS_BALANCE_MIN': ['min'],
            'BB_MONTHS_BALANCE_MAX': ['max'],
            'BB_MONTHS_BALANCE_SIZE': ['mean', 'sum'],

            'ENDDATE_DIFF': ['mean'],
            'AMT_CREDIT_SUM_RATIO': ['mean', 'max'],
            'DAYS_CREDIT_PLAN': ['mean', 'sum'],

            'AMT_CREDIT_DEBT_PERC': ['mean'], #TODO min/maxも効く？
            'AMT_CREDIT_DEBT_DIFF': ['mean','sum']
        }

        bb_aggregations = {
            'MONTHS_BALANCE': ['min', 'max', 'size'],
            'DPD': ['min', 'max', 'mean', 'std']
        }

        # BBをmerge
        bb_agg = self.bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index(['BB_' + e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
        bb_agg.reset_index(inplace=True)

        b = self.bureau
        b = pd.merge(b, bb_agg, how='left', on='SK_ID_BUREAU')

        # カテゴリ別とステータス別に分けておく。
        # TODO: 2年以内など、期間で切る

        agg_active = agg.copy()
        agg_active['AMT_CREDIT_DEBT_PERC_NZ'] = ['min','max']
        agg_active['AMT_CREDIT_DEBT_DIFF_NZ'] = ['sum']

        df = features_common.aggregate(df, agg, b, 'b_')
        df = features_common.aggregate(df, agg_active, b.query('CREDIT_ACTIVE == "Active"'), 'b_active_')
        df = features_common.aggregate(df, agg, b.query('CREDIT_ACTIVE == "Closed"'), 'b_closed_')
        df = features_common.aggregate(df, agg, b.query('CREDIT_TYPE == "Consumer credit"'), 'b_consumer_')
        df = features_common.aggregate(df, agg, b.query('CREDIT_TYPE == "Credit card"'), 'b_credit_')
        df = features_common.aggregate(df, agg, b.query('CREDIT_TYPE == "Car loan"'), 'b_car_')
        df = features_common.aggregate(df, agg, b.query('CREDIT_TYPE == "Mortgage"'), 'b_mortage_')
        df = features_common.aggregate(df, agg, b.query('CREDIT_TYPE == "Microloan"'), 'b_micro_')

        self.bureau = b
        return df
