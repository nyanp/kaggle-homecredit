import pandas as pd
import numpy as np


class Application(object):
    def __init__(self, file=None):
        if file is None:
            self.df = pd.read_feather('../input/application_all.f')
            self.transformed = False
        else:
            self.df = pd.read_feather(file)
            self.transformed = True

    @classmethod
    def from_cache(cls):
        print('app loading from cache...')
        return cls('cache/application.f')

    def fill(self):
        pass

    def transform(self):
        if self.transformed:
            return
        # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
        self.df = self.df[self.df['CODE_GENDER'] != 'XNA']

        # NaN values for DAYS_EMPLOYED: 365.243 -> nan
        self.df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        self.df['CREDIT_TO_ANNUITY_RATIO'] = self.df['AMT_CREDIT'] / self.df['AMT_ANNUITY']
        self.df['CREDIT_TO_GOODS_RATIO'] = self.df['AMT_CREDIT'] / self.df['AMT_GOODS_PRICE']
        self.df['INC_PER_CHLD'] = self.df['AMT_INCOME_TOTAL'] / (1 + self.df['CNT_CHILDREN'])
        self.df['EMPLOY_TO_BIRTH_RATIO'] = self.df['DAYS_EMPLOYED'] / self.df['DAYS_BIRTH']
        self.df['ANNUITY_TO_INCOME_RATIO'] = self.df['AMT_ANNUITY'] / (1 + self.df['AMT_INCOME_TOTAL'])
        self.df['SOURCES_PROD'] = self.df['EXT_SOURCE_1'] * self.df['EXT_SOURCE_2'] * self.df['EXT_SOURCE_3']
        self.df['SOURCES_MEAN'] = self.df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        self.df['SOURCES_STD'] = self.df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        self.df['SOURCES_STD'] = self.df['SOURCES_STD'].fillna(self.df['SOURCES_STD'].mean())
        self.df['CAR_TO_BIRTH_RATIO'] = self.df['OWN_CAR_AGE'] / self.df['DAYS_BIRTH']
        self.df['CAR_TO_EMPLOY_RATIO'] = self.df['OWN_CAR_AGE'] / self.df['DAYS_EMPLOYED']
        self.df['PHONE_TO_BIRTH_RATIO'] = self.df['DAYS_LAST_PHONE_CHANGE'] / self.df['DAYS_BIRTH']
        self.df['PHONE_TO_BIRTH_RATIO_EMPLOYER'] = self.df['DAYS_LAST_PHONE_CHANGE'] / self.df['DAYS_EMPLOYED']
        self.df['CREDIT_TO_INCOME_RATIO'] = self.df['AMT_CREDIT'] / self.df['AMT_INCOME_TOTAL']

        def make_groupby(d_, group_by, target_column, feature_name, aggregation='quantile'):
            df_agg = d_.groupby(group_by)[target_column].agg(aggregation).reset_index()
            df_agg.columns = [group_by, 'tmp']

            d_ = pd.merge(d_, df_agg, on=group_by, how='left')
            d_[feature_name] = d_[target_column] / d_['tmp']
            d_.drop('tmp', axis=1, inplace=True)

            return d_

        # 同一職種内での相対的な収入、借入額、返済額、年齢
        self.df = make_groupby(self.df, 'OCCUPATION_TYPE', 'AMT_INCOME_TOTAL', 'REL_AMT_INCOME_BY_OCCUPATION')
        self.df = make_groupby(self.df, 'OCCUPATION_TYPE', 'AMT_CREDIT', 'REL_AMT_CREDIT_BY_OCCUPATION')
        self.df = make_groupby(self.df, 'OCCUPATION_TYPE', 'AMT_ANNUITY', 'REL_AMT_ANNUITY_BY_OCCUPATION')
        self.df = make_groupby(self.df, 'OCCUPATION_TYPE', 'DAYS_BIRTH', 'REL_DAYS_BIRTH_BY_OCCUPATION')

        self.transformed = True
        self.df.to_feather('cache/application.f')

    def _prev_to_curr_credit_annuity_ratio(self, df, prev):
        prev_ = prev[prev.AMT_ANNUITY > 0]

        # userごとの、前回までの平均と今回のクレジット額の比率
        # CreditとCashで額が大きく違う場合が多いので、それぞれで計算もしておく
        prev_mean = prev_.groupby('SK_ID_CURR')[['AMT_ANNUITY', 'AMT_CREDIT']].mean().reset_index().rename(
            columns={'AMT_ANNUITY': 'MEAN_PREV_ANNUITY', 'AMT_CREDIT': 'MEAN_PREV_CREDIT'})

        prev_mean_cash = prev_[prev_.NAME_CONTRACT_TYPE == 'Cash loans'].groupby('SK_ID_CURR')[
            ['AMT_ANNUITY', 'AMT_CREDIT']].mean().reset_index().rename(
            columns={'AMT_ANNUITY': 'MEAN_PREV_CASH_ANNUITY', 'AMT_CREDIT': 'MEAN_PREV_CASH_CREDIT'})

        prev_mean_revolving = prev_[prev_.NAME_CONTRACT_TYPE == 'Revolving loans'].groupby('SK_ID_CURR')[
            ['AMT_ANNUITY', 'AMT_CREDIT']].mean().reset_index().rename(
            columns={'AMT_ANNUITY': 'MEAN_PREV_REVOLVING_ANNUITY', 'AMT_CREDIT': 'MEAN_PREV_REVOLVING_CREDIT'})

        df_ratio = pd.merge(df[['SK_ID_CURR', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL']], prev_mean,
                            on='SK_ID_CURR', how='left')
        df_ratio = pd.merge(df_ratio, prev_mean_cash, on='SK_ID_CURR', how='left')
        df_ratio = pd.merge(df_ratio, prev_mean_revolving, on='SK_ID_CURR', how='left')

        # df_ratio['PREV_TO_CURR_CREDIT_RATIO'] = df_ratio['AMT_CREDIT'] / df_ratio['MEAN_PREV_CREDIT']

        # TODO: いらないかも
        df_ratio['PREV_TO_CURR_ANNUITY_RATIO'] = df_ratio['AMT_ANNUITY'] / df_ratio['MEAN_PREV_ANNUITY']

        df_ratio['PREV_TO_CURR_CREDIT_RATIO_CASH'] = df_ratio['AMT_CREDIT'] / df_ratio['MEAN_PREV_CASH_CREDIT']
        df_ratio['PREV_TO_CURR_ANNUITY_RATIO_CASH'] = df_ratio['AMT_ANNUITY'] / df_ratio['MEAN_PREV_CASH_ANNUITY']

        df_ratio['PREV_TO_CURR_CREDIT_RATIO_REVOLVING'] = df_ratio['AMT_CREDIT'] / df_ratio[
            'MEAN_PREV_REVOLVING_CREDIT']
        df_ratio['PREV_TO_CURR_ANNUITY_RATIO_REVOLVING'] = df_ratio['AMT_ANNUITY'] / df_ratio[
            'MEAN_PREV_REVOLVING_ANNUITY']

        df_ratio.drop(['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL'], axis=1, inplace=True)

        return pd.merge(df, df_ratio, on='SK_ID_CURR', how='left')

    def _groupby_with_prev(self, df, prev):
        # Organization、Occupationでグルーピングした平均との差
        # Currだけだとデータが少なくて精度があまり出ない

        # currとprevについて、注目する列だけをconcat

        target_columns = ['AMT_ANNUITY', 'AMT_GOODS_PRICE', 'AMT_CREDIT', 'NAME_CONTRACT_TYPE']
        group_columns = ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE']

        df_amt = df[['SK_ID_CURR']+target_columns+group_columns].copy()
        prev_amt = prev[['SK_ID_CURR']+target_columns].copy()

        prev_amt = \
            pd.merge(prev_amt, df[['SK_ID_CURR']+group_columns], on='SK_ID_CURR', how='left')[df_amt.columns]
        prev_amt['is_prev'] = 1

        amt_ttl = pd.concat([df_amt, prev_amt])
        amt_ttl.sort_values(by='SK_ID_CURR', inplace=True)

        def make_groupby(d, by, on_, postfix):
            clm_base = '{}_BY_{}'.format(on_, postfix)

            if clm_base not in d:
                g = d.groupby(by)[on_].mean().reset_index().rename(columns={on_: clm_base})
                d = pd.merge(d, g, on=by, how='left')
            d[clm_base + '_RATIO'] = d[on_] / d[clm_base]
            d[clm_base + '_DIFF'] = d[on_] - d[clm_base]
            # df.drop(clm_base, axis=1, inplace=True)
            return d

        amt_ttl = make_groupby(amt_ttl, ['ORGANIZATION_TYPE', 'NAME_CONTRACT_TYPE'], 'AMT_ANNUITY', 'ORG_CONTRACT')
        amt_ttl = make_groupby(amt_ttl, ['OCCUPATION_TYPE', 'NAME_CONTRACT_TYPE'], 'AMT_ANNUITY', 'ORG_CONTRACT')
        amt_ttl = make_groupby(amt_ttl, ['ORGANIZATION_TYPE', 'NAME_CONTRACT_TYPE'], 'AMT_CREDIT', 'ORG_CONTRACT')
        amt_ttl = make_groupby(amt_ttl, ['OCCUPATION_TYPE', 'NAME_CONTRACT_TYPE'], 'AMT_CREDIT', 'ORG_CONTRACT')
        amt_ttl = make_groupby(amt_ttl, ['ORGANIZATION_TYPE', 'NAME_CONTRACT_TYPE'], 'AMT_GOODS_PRICE', 'ORG_CONTRACT')
        amt_ttl = make_groupby(amt_ttl, ['OCCUPATION_TYPE', 'NAME_CONTRACT_TYPE'], 'AMT_GOODS_PRICE', 'ORG_CONTRACT')

        amt_ttl = amt_ttl[amt_ttl.is_prev.isnull()]
        amt_ttl.drop(['AMT_ANNUITY', 'AMT_GOODS_PRICE', 'AMT_CREDIT', 'NAME_CONTRACT_TYPE',
                      'ORGANIZATION_TYPE', 'OCCUPATION_TYPE', 'is_prev'], axis=1, inplace=True)

        return pd.merge(df, amt_ttl, on='SK_ID_CURR', how='left')

    def transform_with_others(self, df, prev):
        print('transform with others')
        # 他のテーブルとの総合的な特徴
        df = self._prev_to_curr_credit_annuity_ratio(df, prev)

        print(df.shape)
        #
        return self._groupby_with_prev(df, prev)
