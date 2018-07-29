import pandas as pd
import numpy as np
import features_common


class Prev(object):
    def __init__(self, file=None):
        if file is None:
            self.df = pd.read_feather('../input/previous_application.f')
            self.transformed = False
        else:
            self.df = pd.read_feather(file)
            self.transformed = True

    def fill(self):
        if self.transformed:
            return
        self.df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        self.df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        self.df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        self.df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        self.df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    def transform(self):
        if self.transformed:
            return
        self.df['CREDIT_TO_ANNUITY_RATIO'] = self.df['AMT_CREDIT'] / self.df['AMT_ANNUITY']
        self.df['CREDIT_TO_GOODS_RATIO'] = self.df['AMT_GOODS_PRICE'] / self.df['AMT_CREDIT']
        self.df['APP_CREDIT_PERC'] = self.df['AMT_APPLICATION'] / self.df['AMT_CREDIT']
        self.df.to_feather('cache/prev.f')
        self.transformed = True

    @classmethod
    def from_cache(cls):
        print('prev loading from cache...')
        return cls('cache/prev.f')

    def aggregate(self, df_base):
        print('aggregate: prev')

        num_aggregations = {
            'AMT_ANNUITY': ['max', 'mean'],
            'AMT_APPLICATION': ['max', 'mean'],
            'AMT_CREDIT': ['max', 'mean'],
            'APP_CREDIT_PERC': ['max', 'mean'],
            'AMT_DOWN_PAYMENT': ['max', 'mean'],
            'AMT_GOODS_PRICE': ['max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['max', 'mean'],
            'RATE_DOWN_PAYMENT': ['max', 'mean'],
            'DAYS_DECISION': ['max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],

            'CREDIT_TO_ANNUITY_RATIO': ['min', 'max', 'mean'],
            'CREDIT_TO_GOODS_RATIO': ['min', 'max', 'mean']
        }

        p = self.df

        p_approved = p[p.NAME_CONTRACT_STATUS == 'Approved']
        p_refused = p[p.NAME_CONTRACT_STATUS == 'Refused']
        p_cash = p[p.NAME_CONTRACT_TYPE == 'Cash loans']
        p_consumer = p[p.NAME_CONTRACT_TYPE == 'Consumer loans']

        for b, prefix in zip([p, p_approved, p_refused, p_cash, p_consumer],
                             ['p_', 'p_approved_', 'p_refused_', 'p_cash_', 'p_consumer_']):
            agg = b.groupby('SK_ID_CURR').agg(num_aggregations)
            agg.columns = features_common.make_agg_names(prefix, agg.columns.tolist())
            agg.reset_index(inplace=True)
            df_base = pd.merge(df_base, agg, on='SK_ID_CURR', how='left')

        self.df = p

        return df_base
