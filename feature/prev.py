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

    def _target_encoding(self, col, name):
        tgt = self.df.groupby(col)['TARGET'].mean().reset_index()

        tgt.columns = [col, name]
        self.df = pd.merge(self.df, tgt, on=col, how='left')

    def transform(self):
        if self.transformed:
            return
        self.df['CREDIT_TO_ANNUITY_RATIO'] = self.df['AMT_CREDIT'] / self.df['AMT_ANNUITY']
        self.df['CREDIT_TO_GOODS_RATIO'] = self.df['AMT_GOODS_PRICE'] / self.df['AMT_CREDIT']
        self.df['APP_CREDIT_PERC'] = self.df['AMT_APPLICATION'] / self.df['AMT_CREDIT']

        self.df = pd.concat([self.df, pd.get_dummies(self.df['NAME_YIELD_GROUP'], prefix='NAME_YIELD_GROUP')], axis=1)

        self.df = pd.concat([self.df, pd.get_dummies(self.df['NAME_GOODS_CATEGORY'], prefix='NAME_GOODS_CATEGORY')], axis=1)
        self.df = pd.concat([self.df, pd.get_dummies(self.df['PRODUCT_COMBINATION'], prefix='PRODUCT_COMBINATION')], axis=1)

        #app = pd.read_feather('../input/application_all.f')
        #self.df = pd.merge(self.df, app[['SK_ID_CURR', 'TARGET']], on='SK_ID_CURR', how='left')

        # target encoding
        #self._target_encoding('NAME_GOODS_CATEGORY', 'TGT_NAME_GOODS_CATEGORY')
        #self._target_encoding('PRODUCT_COMBINATION', 'TGT_PRODUCT_COMBINATION')

        self.df.to_feather('cache/prev.f')
        self.transformed = True

    @classmethod
    def from_cache(cls):
        print('prev loading from cache...')
        return cls('cache/prev.f')

    def aggregate(self, df_base):
        print('aggregate: prev')

        agg = {
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
            'CREDIT_TO_GOODS_RATIO': ['min', 'max', 'mean'],

            'NAME_YIELD_GROUP_high': ['mean'],
            'NAME_YIELD_GROUP_low_action': ['mean'],
            'NAME_YIELD_GROUP_low_normal': ['mean'],

            'NAME_YIELD_GROUP_middle': ['mean'],
            'NAME_YIELD_GROUP_XNA': ['mean'],

            'NAME_GOODS_CATEGORY_XNA': ['mean'],
            'NAME_GOODS_CATEGORY_Mobile': ['mean'],
            'NAME_GOODS_CATEGORY_Consumer Electronics': ['mean'],
            'NAME_GOODS_CATEGORY_Computers': ['mean'],
        }

        df_base = features_common.aggregate(df_base, agg, self.df, 'p_')
        df_base = features_common.aggregate(df_base, agg, self.df.query('NAME_CONTRACT_STATUS == "Approved"'), 'p_approved')
        df_base = features_common.aggregate(df_base, agg, self.df.query('NAME_CONTRACT_STATUS == "Refused"'), 'p_refused')
        df_base = features_common.aggregate(df_base, agg, self.df.query('NAME_CONTRACT_TYPE == "Cash loans"'), 'p_cash')
        df_base = features_common.aggregate(df_base, agg, self.df.query('NAME_CONTRACT_TYPE == "Consumer loans"'), 'p_cunsumer')



        return df_base
