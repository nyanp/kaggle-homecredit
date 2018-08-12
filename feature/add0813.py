import pandas as pd

df = pd.read_feather('../input/application_all.f')
install = pd.read_feather('../input/installments_payments.csv.f')
bureau = pd.read_feather('../input/bureau.csv.f')
credit = pd.read_feather('../input/credit_card_balance.csv.f')
prev = pd.read_feather('../input/previous_application.csv.f')
pos = pd.read_feather('../input/POS_CASH_balance.csv.f')
bb = pd.read_feather('../input/bureau_balance.csv.f')

install.sort_values(by=['SK_ID_CURR','SK_ID_PREV','DAYS_INSTALMENT'], ascending=False, inplace=True)
prev.sort_values(by=['SK_ID_CURR','SK_ID_PREV','DAYS_DECISION'], ascending=False, inplace=True)
credit.sort_values(by=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'], ascending=False, inplace=True)
pos.sort_values(by=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'], ascending=False, inplace=True)

import numpy as np

df_features = pd.DataFrame()
df_features['SK_ID_CURR'] = df.SK_ID_CURR.unique()

df_features.shape

def already_merged(df_features, df):
    for c in df:
        if c == 'SK_ID_CURR':
            continue
        if c in df_features:
            print('{} is already in df_features. skipped'.format(c))
            return True
    return False

def merge(df_features, df):
    if already_merged(df_features, df):
        return df_features
    df_features = pd.merge(df_features, df, on='SK_ID_CURR', how='left')
    print('merged. shape: {}'.format(df_features.shape))
    return df_features

def make_agg_names(prefix, columns):
    return pd.Index([prefix + e[1] + "(" + e[0] + ")" for e in columns])

def aggregate(df, agg, prefix, by='SK_ID_CURR'):
    df_agg = df.groupby(by).agg(agg)
    df_agg.columns = make_agg_names(prefix, df_agg)
    df_agg.reset_index(inplace=True)
    return df_agg


def completed(pos, credit):
    pos_id = pos.query('NAME_CONTRACT_STATUS == "Completed"').SK_ID_PREV.unique()
    credit_id = credit.query('NAME_CONTRACT_STATUS == "Completed"').SK_ID_PREV.unique()
    return list(pos_id) + list(credit_id)


def active(pos, credit):
    comp = completed(pos, credit)

    pos_id = pos[~pos.SK_ID_PREV.isin(comp)].SK_ID_PREV.unique()
    credit_id = credit[~credit.SK_ID_PREV.isin(comp)].SK_ID_PREV.unique()
    return list(pos_id) + list(credit_id)


prev['DAYS_PLAN'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_FIRST_DUE']
prev['DAYS_PLAN_PER_PAYMENT'] = prev['DAYS_PLAN'] / prev['CNT_PAYMENT']

active_ids = active(pos, credit)

prev_active = prev[prev.SK_ID_PREV.isin(active_ids)]

agg = {
    'CNT_PAYMENT': ['mean'],
    'DAYS_PLAN': ['mean'],
    'DAYS_PLAN_PER_PAYMENT': ['mean']
}

p1 = aggregate(prev_active, agg, 'p_active_')
p2 = aggregate(prev.query('NAME_CONTRACT_TYPE == "Cash loans"'), agg, 'p_cash_')
p3 = aggregate(prev.query('NAME_CONTRACT_TYPE == "Consumer loans"'), agg, 'p_consumer_')
p4 = aggregate(prev.query('NAME_CONTRACT_TYPE == "Revloving loans"'), agg, 'p_credit_')
p5 = aggregate(prev.query('NAME_CONTRACT_TYPE == "Cash loans" and DAYS_DECISION >= -720'), agg, 'p_cash720_')
p6 = aggregate(prev.query('NAME_CONTRACT_TYPE == "Consumer loans" and DAYS_DECISION >= -720'), agg, 'p_consumer720_')
p7 = aggregate(prev.query('NAME_CONTRACT_TYPE == "Revloving loans" and DAYS_DECISION >= -720'), agg, 'p_credit720_')

df_features = merge(df_features, p1)
df_features = merge(df_features, p2)
df_features = merge(df_features, p3)
df_features = merge(df_features, p4)
df_features = merge(df_features, p5)
df_features = merge(df_features, p6)
df_features = merge(df_features, p7)

df_features.to_feather('model/add0813.f')