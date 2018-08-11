import pandas as pd
import os
import sys

def make_agg_names(prefix, columns):
    return pd.Index([prefix + e[1] + "(" + e[0] + ")" for e in columns])


def group_by_1(df, by, target, agg_func, column_name, merge=True):
    if isinstance(by, str):
        by = [by]

    g = df.groupby(by)[target].agg([agg_func]).reset_index()
    g.columns = by + [column_name]

    if merge:
        df = pd.merge(df, g, on=by, how='left')
        return df
    else:
        return g


def aggregate(df, aggregations, target, prefix, key='SK_ID_CURR', count_column=None):
    assert len(df) > 0 and len(target) > 0
    agg = target.groupby(key).agg({**aggregations})
    agg.columns = make_agg_names(prefix, agg.columns.tolist())
    agg.reset_index(inplace=True)
    if count_column is not None:
        agg[count_column] = target.groupby(key).size()
    return pd.merge(df, agg, on=key, how='left')


def extract_active_balance(df, threshold=-12):
    """
    POS_CASH/Credit_Balanceの中から、ActiveなローンのBalanceデータだけを抜いてくる
    最新のMONTHS_BALANCEがthresholdより古いものは、Activeであっても無視

    :param df:
    :param threshold:
    :return:
    """
    prev_id_closed = df[df.NAME_CONTRACT_STATUS == 'Completed'].SK_ID_PREV.unique()
    df_last = df.groupby('SK_ID_PREV')['MONTHS_BALANCE'].max().reset_index()
    prev_id_recent = df_last[df_last['MONTHS_BALANCE'] >= threshold].SK_ID_PREV

    return df[~df.SK_ID_PREV.isin(prev_id_closed) & df.SK_ID_PREV.isin(prev_id_recent)].reset_index(drop=True)


def read_csv(file):
    if os.path.exists(file + '.f'):
        return pd.read_feather(file + '.f')
    else:
        df = pd.read_csv(file)
        df.to_feather(file + '.f')
        return df

def read_application():
    if os.path.exists('../input/application_all.f'):
        return pd.read_feather('../input/application_all.f')
    else:
        df_train = read_csv('../input/application_train.csv')
        df_test = read_csv('../input/application_test.csv')

        df_all = pd.concat([df_train,df_test]).reset_index(drop=True)
        df_all.to_feather('../input/application_all.f')

        return df_all
