import pandas as pd
import features_common


class PosCash(object):
    def __init__(self):
        self.pos = pd.read_feather('../input/POS_CASH_balance.f')
        self.pos.sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'], ascending=False, inplace=True)

    def fill(self):
        pass

    def transform(self):
        pass

    def aggregate(self, df):
        print('aggregate: cash')

        num_aggregations = {
            'MONTHS_BALANCE': ['count', 'mean'],
            'SK_DPD': ['sum', 'max'],
            'SK_DPD_DEF': ['sum', 'max'],
            # 'SK_DPD_BINARY': ['sum']
        }

        pos_agg = self.pos.groupby('SK_ID_CURR').agg({**num_aggregations})
        pos_agg.columns = features_common.make_agg_names('pos_', pos_agg.columns.tolist())
        pos_agg.reset_index(inplace=True)

        df = pd.merge(df, pos_agg, on='SK_ID_CURR', how='left')

        # additional
        # CNT_INSTALMENT_AHEAD_RATIO:前倒しの割合
        # CNT_INSTALMENT_AHEAD:前倒しの月数

        # TODO: DPDやDPD_DEFも、期間を切ってみる
        def calc_aheads(df):
            tail = df.groupby('SK_ID_PREV')['SK_ID_CURR', 'SK_ID_PREV', 'CNT_INSTALMENT'].tail(1)
            head = df.groupby('SK_ID_PREV')['SK_ID_CURR', 'SK_ID_PREV', 'CNT_INSTALMENT'].head(1)
            tail.rename(columns={'CNT_INSTALMENT': 'PLANNED_CNT_INSTALMENT'}, inplace=True)
            head.rename(columns={'CNT_INSTALMENT': 'ACTUAL_CNT_INSTALMENT'}, inplace=True)
            tail = pd.merge(tail, head[['SK_ID_PREV', 'ACTUAL_CNT_INSTALMENT']], on='SK_ID_PREV', how='left')
            tail['CNT_INSTALMENT_AHEAD'] = tail['PLANNED_CNT_INSTALMENT'] - tail['ACTUAL_CNT_INSTALMENT']
            tail['CNT_INSTALMENT_AHEAD_RATIO'] = tail['ACTUAL_CNT_INSTALMENT'] / tail['PLANNED_CNT_INSTALMENT']
            return tail

        pos_prev = calc_aheads(self.pos)
        pos_prev12 = calc_aheads(self.pos[self.pos.MONTHS_BALANCE >= -12])
        pos_prev12.head()

        pos_agg = pos_prev.groupby('SK_ID_CURR').agg({
            'CNT_INSTALMENT_AHEAD': ['min', 'max'],
            'CNT_INSTALMENT_AHEAD_RATIO': ['min']
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

        df = pd.merge(df, pos_all, on='SK_ID_CURR', how='left')

        return df
