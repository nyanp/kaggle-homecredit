import pandas as pd
import numpy as np

df = pd.read_feather('input/application_all.f')
install = pd.read_feather('input/installments_payments.csv.f')
bureau = pd.read_feather('input/bureau.csv.f')
credit = pd.read_feather('input/credit_card_balance.csv.f')
prev = pd.read_feather('input/previous_application.csv.f')
pos = pd.read_feather('input/POS_CASH_balance.csv.f')
bb = pd.read_feather('input/bureau_balance.csv.f')

install.sort_values(by=['SK_ID_CURR','SK_ID_PREV','DAYS_INSTALMENT'], ascending=False, inplace=True)
prev.sort_values(by=['SK_ID_CURR','SK_ID_PREV','DAYS_DECISION'], ascending=False, inplace=True)
credit.sort_values(by=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'], ascending=False, inplace=True)
pos.sort_values(by=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'], ascending=False, inplace=True)

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

df_features = merge(df_features, df[['SK_ID_CURR','TARGET']])

install['DPD'] = install['DAYS_ENTRY_PAYMENT'] - install['DAYS_INSTALMENT']
install['DBD'] = install['DAYS_INSTALMENT'] - install['DAYS_ENTRY_PAYMENT']
install['DPD'] = install['DPD'].apply(lambda x: x if x > 0 else 0)
install['DBD'] = install['DBD'].apply(lambda x: x if x > 0 else 0)

def dd_first(install, n, prefix):
    ins = install[install.NUM_INSTALMENT_NUMBER <= n]
    iagg = ins.groupby('SK_ID_CURR').agg({
        'DPD':['mean'],
        'DBD':['mean']
    })
    iagg.columns = make_agg_names(prefix, iagg)
    iagg.reset_index(inplace=True)
    return iagg

df_features = merge(df_features, dd_first(install, 3, 'ins_first3_'))
df_features = merge(df_features, dd_first(install, 5, 'ins_first5_'))
df_features = merge(df_features, dd_first(install, 10, 'ins_first10_'))

install['DPD_diff'] = install['DPD'] - install.groupby('SK_ID_PREV')['DPD'].shift(-1)
install['DPD_diff2'] = install['DPD_diff'] - install.groupby('SK_ID_PREV')['DPD_diff'].shift(-1)

agg = {
    'DPD_diff':['mean','max'],
    'DPD_diff2':['mean','max']
}

iagg = aggregate(install, agg, 'ins_')

df_features = merge(df_features, iagg)

pos['DPD_diff'] = pos['SK_DPD'] - pos.groupby('SK_ID_PREV')['SK_DPD'].shift(-1)
pos['DPD_diff2'] = pos['DPD_diff'] - pos.groupby('SK_ID_PREV')['DPD_diff'].shift(-1)

agg = {
    'DPD_diff':['mean','max'],
    'DPD_diff2':['mean','max']
}

pagg = aggregate(pos, agg, 'pos_')
df_features = merge(df_features, pagg)

pagg = aggregate(pos.query('MONTHS_BALANCE >= -24'), agg, 'pos24_')
df_features = merge(df_features, pagg)

agg = {
    'AMT_CREDIT': ['var','skew']
}
pagg = aggregate(pos, agg, 'pos_')
df_features = merge(df_features, pagg)

prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

prev['CREDIT_TO_ANNUITY_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
prev['CREDIT_TO_GOODS_RATIO'] = prev['AMT_GOODS_PRICE'] / prev['AMT_CREDIT']
prev['NAME_CONTRACT_STATUS_Approved'] = (prev['NAME_CONTRACT_STATUS'] == 'Approved').astype(np.int32)
prev['NAME_CONTRACT_STATUS_Refused'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(np.int32)

agg = {
    'AMT_ANNUITY': ['count', 'mean', 'count'],
    'DAYS_DECISION': ['max','mean'],
    'CREDIT_TO_GOODS_RATIO': ['mean'],
    'CNT_PAYMENT': ['mean'],
    'NAME_CONTRACT_STATUS_Approved':['mean'],
    'NAME_CONTRACT_STATUS_Refused':['mean']
}

p1 = aggregate(prev.query('NAME_PRODUCT_TYPE == "x-sell"'), agg, 'p_xsell_')
p2 = aggregate(prev.query('NAME_PRODUCT_TYPE == "walk-in"'), agg, 'p_walkin_')

df_features = merge(df_features, p1)
df_features = merge(df_features, p2)

df_features.drop('TARGET',axis=1).to_feather('model/add0815.f')
