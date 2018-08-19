import pandas as pd
import numpy as np

df = pd.read_feather('../input/application_all.f')
install = pd.read_feather('../input/installments_payments.csv.f')
prev = pd.read_feather('../input/previous_application.csv.f')
pos = pd.read_feather('../input/POS_CASH_balance.csv.f')
bureau = pd.read_feather('../input/bureau.csv.f')
bb =pd.read_feather('../input/bureau_balance.csv.f')

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
bureau['AMT_CREDIT_SUM'].replace(0, np.nan, inplace=True)
bureau['AMT_CREDIT_SUM_DEBT'].replace(0, np.nan, inplace=True)

df_features = pd.DataFrame()
df_features['SK_ID_CURR'] = df.SK_ID_CURR.unique()

df_features = merge(df_features, df[['SK_ID_CURR', 'TARGET']])

agg = {
    'AMT_CREDIT_SUM': ['sum']
}

b1 = aggregate(bureau.query('CREDIT_TYPE == "Consumer credit" and CREDIT_ACTIVE == "Closed"'), agg, 'b_closed_consumer_')
b2 = aggregate(bureau.query('CREDIT_TYPE == "Credit card" and CREDIT_ACTIVE == "Closed"'), agg, 'b_closed_credit_')
b3 = aggregate(bureau.query('CREDIT_TYPE == "Car loan" and CREDIT_ACTIVE == "Closed"'), agg, 'b_closed_car_')
b4 = aggregate(bureau.query('CREDIT_TYPE == "Mortgage" and CREDIT_ACTIVE == "Closed"'), agg, 'b_closed_mortage_')
b5 = aggregate(bureau.query('CREDIT_TYPE == "Microloan" and CREDIT_ACTIVE == "Closed"'), agg, 'b_closed_micro_')

df_features = merge(df_features, b1)
df_features = merge(df_features, b2)
df_features = merge(df_features, b3)
df_features = merge(df_features, b4)
df_features = merge(df_features, b5)

def recency(df, name):
    agg = df.groupby('SK_ID_CURR')['DAYS_ENTRY_PAYMENT'].max().reset_index()
    agg.columns = ['SK_ID_CURR', name]
    return agg

i1 = recency(install.query('DPD > 0'), 'ins_RECENCY(DPD > 0)')
i2 = recency(install.query('DBD > 0'), 'ins_RECENCY(DBD > 0)')

df_features = merge(df_features, i1)
df_features = merge(df_features, i2)


def quantized_dpd(install, col, n, q):
    install['q' + col] = ((install[col] + n - 1) // n).clip(0, q).astype('category')

    onehot = pd.get_dummies(install['q' + col])
    onehot.columns = ['q{}_{}'.format(col, i) for i in range(q + 1)]

    agg = {
        c: ['mean'] for c in onehot
    }

    onehot = pd.concat([install[['SK_ID_CURR']], onehot], axis=1)

    return aggregate(onehot, agg, 'ins_')


i1 = quantized_dpd(install, 'DPD', 15, 10)
i2 = quantized_dpd(install, 'DBD', 15, 10)

df_features = merge(df_features, i1)
df_features = merge(df_features, i2)

bb_ = pd.merge(bb, bureau[['SK_ID_CURR','SK_ID_BUREAU']], on='SK_ID_BUREAU', how='left')
bb_ = pd.concat([bb_, pd.get_dummies(bb_[['STATUS']])], axis=1)
bb_.head()

agg = {
    'STATUS_C':['mean'],
    'STATUS_0':['mean'],
    'STATUS_1':['mean'],
    'STATUS_2':['mean'],
    'STATUS_3':['mean'],
    'STATUS_4':['mean'],
    'STATUS_5':['mean'],
    'STATUS_X':['mean']
}
agg2 = {
    'STATUS_0':['mean'],
    'STATUS_1':['mean'],
    'STATUS_2':['mean'],
    'STATUS_3':['mean'],
    'STATUS_4':['mean'],
    'STATUS_5':['mean']
}

b1 = aggregate(bb_, agg, 'b_bb_')
b2 = aggregate(bb_.query('STATUS != "X" and STATUS != "C"'), agg2, 'b_bb_wo_CX_')

df_features = merge(df_features, b1)
df_features = merge(df_features, b2)

bureau['DAYS_CREDIT_PLAN'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT']
bureau['CURRENT_PROGRESS_PERC'] = (1.0 - bureau['DAYS_CREDIT_ENDDATE'] / bureau['DAYS_CREDIT_PLAN']).clip(0.0, 1.0)
bureau['EST_AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM'] * (1.0 - bureau['CURRENT_PROGRESS_PERC'])
bureau['CREDIT_DEBT_ESTIMATED_VS_ACTUAL'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['EST_AMT_CREDIT_SUM_DEBT']

agg = {
    'CURRENT_PROGRESS_PERC': ['mean', 'min', 'max'],
    'EST_AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
    'CREDIT_DEBT_ESTIMATED_VS_ACTUAL': ['mean', 'min', 'max']
}

b1 = aggregate(bureau.query('CREDIT_TYPE == "Consumer credit"'), agg, 'b_consumer_')
b2 = aggregate(bureau.query('CREDIT_TYPE == "Consumer credit" and CREDIT_ACTIVE == "Active"'), agg, 'b_consumer_active_')

df_features = merge(df_features, b1)
df_features = merge(df_features, b2)

agg = {
    'MONTHS_BALANCE':['std']
}

b1 = aggregate(bb_, agg, 'b_bb_')

df_features = merge(df_features, b1)

df_features.to_feather('../model/add0819.f')
