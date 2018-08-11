import pandas as pd
import features_common
import numpy as np


class Install(object):
    def __init__(self, file=None):
        if file is None:
            self.df = features_common.read_csv('../input/installments_payments.csv')

            # 最近の支払ほど上に。
            self.df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_ENTRY_PAYMENT'], ascending=False, inplace=True)
            self.df.reset_index(inplace=True, drop=True)

            self.transformed = False
        else:
            self.df = pd.read_feather(file)
            self.transformed = True

        prev = features_common.read_csv('../input/previous_application.csv')
        prev = prev[['SK_ID_PREV','NAME_CONTRACT_TYPE']]
        self.df = pd.merge(self.df, prev, on='SK_ID_PREV', how='left')
        del prev

        credit = features_common.read_csv('../input/credit_card_balance.csv')
        credit_ids = credit.SK_ID_PREV.unique()
        self.df['is_credit'] = self.df.SK_ID_PREV.isin(credit_ids).astype(np.int32)
        del credit

    @classmethod
    def from_cache(cls):
        print('install loading from cache...')
        return cls('cache/install.f')

    def fill(self):
        pass

    def transform(self):
        if self.transformed:
            return

        self._transform_per_payment()

        ins = self.df

        ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
        # Days past due and days before due (no negative values)
        ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
        ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
        ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

        # [採用] DPDの条件分け
        # 同一のSK_ID_PREV/NUM_INSTALMENT_NUMBERに対して、複数回の支払が記録されていることがある。
        # これは一部のみ先行して支払を行い、その後期限切れで残りの支払を行った分だと考えられる。そこで、遅延を
        # - 期限までに一度も支払わず、期限後に支払った (DPD_without_prev)
        # - 期限までに1度以上支払い、期限後に残りを支払った (DPD_with_prev)
        # の2つに分解する。

        # 各支払予定について、初回の支払が期限前だったかどうか
        ins['DBD_1st'] = (ins['FIRST_PAYMENT'] < ins['DAYS_INSTALMENT'])

        ins['DPD_without_prev'] = ins['DPD'] * (~ins['DBD_1st']).astype(np.int32)
        ins['DPD_with_prev'] = ins['DPD'] * (ins['DBD_1st']).astype(np.int32)

        # [採用]遅延日数×遅延しなかった支払の割合
        # TODO: 遅延した支払の割合のほうが良いのでは？
        g = ins[ins.DPD == 0].groupby(['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER', 'NUM_INSTALMENT_VERSION'])[
            'AMT_PAYMENT'].sum().reset_index()
        g.columns = ['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER', 'NUM_INSTALMENT_VERSION', 'AMT_PAYMENT_SUM_ON_SCHEDULE']
        ins = pd.merge(ins, g, on=['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER', 'NUM_INSTALMENT_VERSION'], how='left')
        ins['DPD_RATIO'] = (ins['DPD'] > 0).astype(np.int32) * ins['AMT_PAYMENT_SUM_ON_SCHEDULE'] / ins['AMT_PAYMENT']
        ins['DPD_RATIO'] = ins['DPD_RATIO'].fillna(0)

        # [採用]支払の間隔
        ins['DAYS_ENTRY_PAYMENT_INTERVAL'] = ins['DAYS_ENTRY_PAYMENT'] - ins.groupby('SK_ID_PREV')[
            'DAYS_ENTRY_PAYMENT'].shift(-1)

        self.df = ins

        self.df.to_feather('cache/install.f')
        self.transformed = True

    def _transform_per_payment(self):
        # paymentごとの特徴量

        # 1回の支払予定を何回に分割して支払っているか
        self.df = features_common.group_by_1(self.df,
                                             ['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER', 'NUM_INSTALMENT_VERSION'],
                                                  'AMT_PAYMENT',
                                                  'count',
                                                  'N_PAYMENTS')

        # 最初の支払日。単独では使わない
        self.df = features_common.group_by_1(self.df,
                                             ['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER', 'NUM_INSTALMENT_VERSION'],
                                                  'DAYS_ENTRY_PAYMENT',
                                                  'min',
                                                  'FIRST_PAYMENT')

    def aggregate(self, df_base):
        print('aggregate: install')

        # full period
        agg_full = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum', 'min', 'std'],
            'DBD': ['max', 'mean', 'sum', 'min', 'std'],
            'PAYMENT_PERC': ['max', 'mean', 'var', 'min', 'std'],
            'PAYMENT_DIFF': ['max', 'mean', 'var', 'min', 'std'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum', 'min', 'std'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum', 'std'],

            'DPD_with_prev': ['min'],
            'DPD_without_prev': ['min'],
            'DPD_RATIO': ['min', 'mean'],
            'N_PAYMENTS': ['max']
        }

        # within 365days
        agg_365 = {
            'AMT_PAYMENT': ['sum'],
            'PAYMENT_PERC': ['mean', 'min', 'var', 'std'],
            'AMT_INSTALMENT': ['max', 'sum', 'std'],
            'PAYMENT_DIFF': ['max', 'mean', 'var', 'std'],

            'DPD': ['max', 'mean', 'sum', 'std'],
            'DBD': ['mean'],
            'DAYS_ENTRY_PAYMENT_INTERVAL': ['mean', 'min'],
            'DAYS_ENTRY_PAYMENT': ['mean'],
        }

        # within 180days
        agg_180 = {
            'AMT_PAYMENT': ['sum', 'min', 'max', 'std', 'mean'],
            'PAYMENT_PERC': ['mean', 'min', 'var', 'std'],
            'AMT_INSTALMENT': ['mean', 'sum'],
            'PAYMENT_DIFF': ['max', 'mean'],

            'DPD': ['max', 'mean', 'sum', 'std'],
            'DBD': ['max', 'mean', 'sum', 'std'],
            # 'DAYS_ENTRY_PAYMENT_INTERVAL': ['mean', 'min'], # TODO: 365で効いて180で効かない＝prev-id同士のintervalが大事？
            'DAYS_ENTRY_PAYMENT': ['sum'],
        }

        df_base = features_common.aggregate(df_base, agg_full, self.df, 'ins_')

        df_base = features_common.aggregate(df_base, agg_full, self.df.query('NAME_CONTRACT_TYPE == "Consumer loans"'), 'ins_consumer_')
        df_base = features_common.aggregate(df_base, agg_full, self.df.query('NAME_CONTRACT_TYPE == "Cash loans"'), 'ins_cash_')
        df_base = features_common.aggregate(df_base, agg_full, self.df.query('is_credit == 1'), 'ins_credit_')

        # note: 720, 90daysを足してもスコア上がらず。countを足すのもダメ。
        df_base = features_common.aggregate(df_base, agg_365, self.df.query('DAYS_ENTRY_PAYMENT >= -365'), 'ins365_')
        df_base = features_common.aggregate(df_base, agg_180, self.df.query('DAYS_ENTRY_PAYMENT >= -180'), 'ins180_')

        return df_base
