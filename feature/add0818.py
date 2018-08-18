import pandas as pd
import numpy as np

df = pd.read_feather('../input/application_all.f')
install = pd.read_feather('../input/installments_payments.csv.f')
prev = pd.read_feather('../input/previous_application.csv.f')
pos = pd.read_feather('../input/POS_CASH_balance.csv.f')

install.sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT', 'NUM_INSTALMENT_NUMBER'], ascending=False, inplace=True)
prev.sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_DECISION'], ascending=False, inplace=True)
pos.sort_values(by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'], ascending=False, inplace=True)

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

install['DPD'] = install['DAYS_ENTRY_PAYMENT'] - install['DAYS_INSTALMENT']
install['DBD'] = install['DAYS_INSTALMENT'] - install['DAYS_ENTRY_PAYMENT']
install['DPD'] = install['DPD'].apply(lambda x: x if x > 0 else 0)
install['DBD'] = install['DBD'].apply(lambda x: x if x > 0 else 0)

df_features = pd.DataFrame()
df_features['SK_ID_CURR'] = df.SK_ID_CURR.unique()

df_features = merge(df_features, df[['SK_ID_CURR', 'TARGET']])

tmp = install.groupby(['SK_ID_PREV','DAYS_ENTRY_PAYMENT'])['AMT_PAYMENT'].agg(['sum','count']).reset_index()
tmp.columns = ['SK_ID_PREV','DAYS_ENTRY_PAYMENT','AMT_PAYMENT_DAY','N_PAYMENT_DAY']
tmp = pd.merge(tmp, prev[['SK_ID_PREV','AMT_ANNUITY','NAME_CONTRACT_TYPE']], on='SK_ID_PREV', how='left')

install_ = pd.merge(install, tmp, on=['SK_ID_PREV','DAYS_ENTRY_PAYMENT'], how='left')

dbd_in_prev_ins = install_.query('N_PAYMENT_DAY > 1 and DBD > 0')
dbd_not_in_prev = install_.query('N_PAYMENT_DAY == 1 and DBD > 0')

agg = {
    'DBD':['std','mean']
}

i1 = aggregate(dbd_not_in_prev.query('NAME_CONTRACT_TYPE == "Revolving loans"'), agg, 'ins_not_in_prev_credit_')
i2 = aggregate(dbd_in_prev_ins.query('NAME_CONTRACT_TYPE == "Revolving loans"'), agg, 'ins_in_prev_credit_')
i3 = aggregate(dbd_not_in_prev.query('NAME_CONTRACT_TYPE == "Consumer loans"'), agg, 'ins_not_in_prev_consumer_')
i4 = aggregate(dbd_in_prev_ins.query('NAME_CONTRACT_TYPE == "Consumer loans"'), agg, 'ins_in_prev_consumer_')
i5 = aggregate(dbd_not_in_prev.query('NAME_CONTRACT_TYPE == "Cash loans"'), agg, 'ins_not_in_prev_cash_')
i6 = aggregate(dbd_in_prev_ins.query('NAME_CONTRACT_TYPE == "Cash loans"'), agg, 'ins_in_prev_cash_')

df_features = merge(df_features, i1)
df_features = merge(df_features, i2)
df_features = merge(df_features, i3)
df_features = merge(df_features, i4)
df_features = merge(df_features, i5)
df_features = merge(df_features, i6)

install_['PAYMENT_TO_ANNUITY_RATIO'] = install_['AMT_PAYMENT_DAY'] / install_['AMT_ANNUITY']

iagg = install_.drop_duplicates(subset=['SK_ID_PREV','DAYS_ENTRY_PAYMENT']).groupby('SK_ID_CURR').agg({
    'PAYMENT_TO_ANNUITY_RATIO':['max','mean'],
    'AMT_PAYMENT_DAY':['max','mean']
})
iagg.columns = make_agg_names('ins_', iagg)
iagg.reset_index(inplace=True)

df_features = merge(df_features, iagg)

tmp = install[install.DBD > 0].groupby('SK_ID_PREV')['AMT_PAYMENT'].sum().reset_index()
tmp.columns = ['SK_ID_PREV','AMT_PAYMENT_DBD']

tmp2 = install[install.DBD == 0].groupby('SK_ID_PREV')['AMT_PAYMENT'].sum().reset_index()
tmp2.columns = ['SK_ID_PREV','AMT_PAYMENT_NO_DBD']

iagg = install.groupby(['SK_ID_CURR','SK_ID_PREV'])['AMT_PAYMENT'].sum().reset_index()
iagg = pd.merge(iagg, tmp, on='SK_ID_PREV', how='left')
iagg = pd.merge(iagg, tmp2, on='SK_ID_PREV', how='left')
iagg = iagg.fillna(0)
iagg['DBD_RATIO'] = iagg['AMT_PAYMENT_DBD'] / iagg['AMT_PAYMENT']

iagg = iagg.groupby(['SK_ID_CURR']).agg({'DBD_RATIO':['mean','max','min']})
iagg.columns = make_agg_names('ins_', iagg)
iagg.reset_index(inplace=True)

df_features = merge(df_features, iagg)

install_ = pd.merge(install, prev[['SK_ID_PREV','AMT_ANNUITY']], on='SK_ID_PREV', how='left').rename(columns={'AMT_ANNUITY':'AMT_ANNUITY_PREV'})
install_ = pd.merge(install_, df[['SK_ID_CURR','AMT_ANNUITY']], on='SK_ID_CURR', how='left')
install_['ANNUITY_RATIO'] = install_['AMT_ANNUITY_PREV'] / install_['AMT_ANNUITY']

similar_install = install_.query('ANNUITY_RATIO > 0.5 and ANNUITY_RATIO < 2.0')
similar_install = pd.merge(similar_install, prev[['SK_ID_PREV','NAME_CONTRACT_TYPE','DAYS_DECISION']], on='SK_ID_PREV', how='left')

similar_install2 = install_.query('ANNUITY_RATIO > 0.7 and ANNUITY_RATIO < 1.3')
similar_install2 = pd.merge(similar_install2, prev[['SK_ID_PREV','NAME_CONTRACT_TYPE','DAYS_DECISION']], on='SK_ID_PREV', how='left')

similar_install3 = install_.query('ANNUITY_RATIO > 0.3 and ANNUITY_RATIO < 1.3')
similar_install3 = pd.merge(similar_install3, prev[['SK_ID_PREV','NAME_CONTRACT_TYPE','DAYS_DECISION']], on='SK_ID_PREV', how='left')

agg = {
    'DBD':['mean','sum','std'],
    'DPD':['mean','std']
}

i1 = aggregate(similar_install.query('DAYS_ENTRY_PAYMENT >= -365'), agg, 'ins365_similar_')
i2 = aggregate(similar_install, agg, 'ins_similar_')
i3 = aggregate(similar_install2.query('DAYS_ENTRY_PAYMENT >= -365'), agg, 'ins365_similar2_')
i4 = aggregate(similar_install2, agg, 'ins_similar_')
i5 = aggregate(similar_install3.query('DAYS_ENTRY_PAYMENT >= -365'), agg, 'ins365_similar3_')
i6 = aggregate(similar_install3, agg, 'ins_similar_')

df_features = merge(df_features, i1)
df_features = merge(df_features, i2)

install_ = install.copy()
install_['NUMBER_DIFF'] = install_['NUM_INSTALMENT_NUMBER'] - install_.groupby(['SK_ID_PREV','NUM_INSTALMENT_VERSION'])['NUM_INSTALMENT_NUMBER'].shift(-1)

iagg = aggregate(install_, {'NUMBER_DIFF':['mean','std','min']}, 'ins_')
df_features = merge(df_features, iagg)

dbd = install.groupby(['SK_ID_CURR','SK_ID_PREV'])['DBD'].agg(['min','max']).reset_index()
dbd.columns = ['SK_ID_CURR','SK_ID_PREV', 'DBD_MIN','DBD_MAX']
dbd['DBD_DIFF'] = dbd['DBD_MAX'] - dbd['DBD_MIN']

iagg = aggregate(dbd, {'DBD_DIFF':['mean']}, 'ins_')
df_features = merge(df_features, iagg)

df_features.to_feather('../model/add0818.f')
