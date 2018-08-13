import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import time
import gc

from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

with timer('load dataframes'):
    # base
    df = pd.read_feather('../input/application_all.f')

    # 信用機関からの信用情報。ユーザーごとに、過去の借入に関する情報が記録されている
    bureau = pd.read_feather('../input/bureau.csv.f')

    bureau_balance = pd.read_feather('../input/bureau_balance.csv.f')

    pos = pd.read_feather('../input/POS_CASH_balance.csv.f')

    credit = pd.read_feather('../input/credit_card_balance.csv.f')

    # 過去のローン申し込み情報
    prev = pd.read_feather('../input/previous_application.csv.f')

    # 過去の分割払い(install)に関する支払情報
    install = pd.read_feather('../input/installments_payments.csv.f')

def make_agg_names(prefix, columns):
    return pd.Index([prefix + e[1] + "(" + e[0] + ")" for e in columns])

aggregations = []

# 信用情報が更新されていないものは、ENDDATEが負なのにActiveなものがある。情報が古いだけなので、そのような行はActive扱いとしない。
b_active = bureau.query('CREDIT_ACTIVE == "Active" and DAYS_CREDIT_ENDDATE >= 0')

b_mean_enddate = b_active.query('CREDIT_TYPE == "Consumer credit"').groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].mean().reset_index()
b_mean_enddate.columns = ['SK_ID_CURR', 'b_active_consumer_mean(DAYS_CREDIT_ENDDATE)']

b_mean_enddate2 = b_active.query('CREDIT_TYPE == "Credit card"').groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].mean().reset_index()
b_mean_enddate2.columns = ['SK_ID_CURR', 'b_active_credit_mean(DAYS_CREDIT_ENDDATE)']

aggregations.append(b_mean_enddate)
aggregations.append(b_mean_enddate2)

# 情報が古すぎると残り返済額が信頼できなくなる。
b_amt = b_active.query('CREDIT_TYPE == "Consumer credit" and AMT_CREDIT_SUM_DEBT > 0 and DAYS_CREDIT_UPDATE > -90')
b_sum_cons = b_amt.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum().reset_index()
b_sum_cons.columns = ['SK_ID_CURR','SUM(AMT_DEBT_ACTIVE_LOAN_BUREAU_CONSUMER)']

b_amt = b_active.query('CREDIT_TYPE == "Credit card" and AMT_CREDIT_SUM_DEBT > 0 and DAYS_CREDIT_UPDATE > -90')
b_sum_credit = b_amt.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum().reset_index()
b_sum_credit.columns = ['SK_ID_CURR','SUM(AMT_DEBT_ACTIVE_LOAN_BUREAU_CREDIT)']

print(b_amt.shape)

aggregations.append(b_sum_cons)
aggregations.append(b_sum_credit)

prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
prev['DAYS_FIRST_DUE'].replace(0, np.nan, inplace=True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(0, np.nan, inplace=True)

prev['DAYS_CREDIT_PLAN'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_FIRST_DUE']

p_mean_plan = prev[prev.NAME_CONTRACT_TYPE.isin(['Cash loans','Consumer loans'])].groupby('SK_ID_CURR')['DAYS_CREDIT_PLAN'].mean().reset_index()
p_mean_plan.columns = ['SK_ID_CURR','p_consumer_mean(DAYS_CREDIT_PLAN)']

aggregations.append(p_mean_plan)

prev_cash = prev[prev.NAME_CONTRACT_TYPE.isin(['Cash loans','Consumer loans'])]
prev_credit = prev[prev.NAME_CONTRACT_TYPE == 'Revolving loans']

p_cash_amt = prev_cash.query('DAYS_DECISION >= -720').groupby('SK_ID_CURR')['AMT_CREDIT'].sum().reset_index()
p_cash_amt.columns = ['SK_ID_CURR','p_pos_720_sum(AMT_CREDIT_SUM)']

p_credit_amt = prev_credit.query('DAYS_DECISION >= -720').groupby('SK_ID_CURR')['AMT_CREDIT'].sum().reset_index()
p_credit_amt.columns = ['SK_ID_CURR','p_credit_720_sum(AMT_CREDIT_SUM)']

aggregations.append(p_cash_amt)
aggregations.append(p_credit_amt)

bureau['ENDDATE_DIFF'] = bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT_ENDDATE']
bureau['AHEAD_RATIO'] = (bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT']) / (bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT'])

bureau_ = bureau[bureau.DAYS_CREDIT_ENDDATE > bureau.DAYS_CREDIT]
print('{} -> {}'.format(len(bureau), len(bureau_)))

b_ahead_max = bureau_.query('CREDIT_ACTIVE == "Closed" and CREDIT_TYPE == "Consumer credit"').groupby('SK_ID_CURR')['ENDDATE_DIFF'].max().reset_index()
b_ahead_max.columns = ['SK_ID_CURR','b_consumer_max(ENDDATE_DIFF)']

aggregations.append(b_ahead_max)

b_ahead_ratio_mean = bureau_.query('CREDIT_ACTIVE == "Closed" and CREDIT_TYPE == "Consumer credit"').groupby('SK_ID_CURR')['ENDDATE_DIFF'].mean().reset_index()
b_ahead_ratio_mean.columns = ['SK_ID_CURR','b_consumer_mean(AHEAD_RATIO)']

b_ahead_ratio_720_mean = bureau_.query('CREDIT_ACTIVE == "Closed" and CREDIT_TYPE == "Consumer credit" and DAYS_CREDIT >= -720').groupby('SK_ID_CURR')['ENDDATE_DIFF'].mean().reset_index()
b_ahead_ratio_720_mean.columns = ['SK_ID_CURR','b_consumer720_mean(AHEAD_RATIO)']

aggregations.append(b_ahead_ratio_720_mean)

bb_active = bureau_balance[bureau_balance.STATUS != 'C']
bb = pd.merge(bb_active, bureau, on='SK_ID_BUREAU', how='left')

bb_cash = bb.query('CREDIT_TYPE == "Consumer credit"')
bb_credit = bb.query('CREDIT_TYPE == "Credit card"')

bb_cash.head()

bb_cash_mb_mean = bb_cash.groupby('SK_ID_CURR')['MONTHS_BALANCE'].mean().reset_index()
bb_cash_mb_mean.columns = ['SK_ID_CURR', 'b_cash_mean(MONTHS_BALANCE)']

bb_cash365_mb_mean = bb_cash.query('DAYS_CREDIT >= -365').groupby('SK_ID_CURR')['MONTHS_BALANCE'].mean().reset_index()
bb_cash365_mb_mean.columns = ['SK_ID_CURR', 'b_cash365_mean(MONTHS_BALANCE)']

bb_credit_mb_mean = bb_credit.groupby('SK_ID_CURR')['MONTHS_BALANCE'].mean().reset_index()
bb_credit_mb_mean.columns = ['SK_ID_CURR', 'b_credit_mean(MONTHS_BALANCE)']

bb_credit365_mb_mean = bb_credit.query('DAYS_CREDIT >= -365').groupby('SK_ID_CURR')['MONTHS_BALANCE'].mean().reset_index()
bb_credit365_mb_mean.columns = ['SK_ID_CURR', 'b_credit365_mean(MONTHS_BALANCE)']

aggregations.append(bb_cash_mb_mean)
aggregations.append(bb_cash365_mb_mean)
aggregations.append(bb_credit_mb_mean)
aggregations.append(bb_credit365_mb_mean)

pos_closed_ids = pos[pos.NAME_CONTRACT_STATUS == 'Completed'].SK_ID_PREV.unique()
credit_closed_ids = credit[credit.NAME_CONTRACT_STATUS == 'Completed'].SK_ID_PREV.unique()
all_closed_ids = list(pos_closed_ids)+list(credit_closed_ids)

prev_closed = prev[prev.SK_ID_PREV.isin(all_closed_ids)]
len(prev_closed)

p_closed = prev_closed.groupby('SK_ID_CURR')['AMT_CREDIT'].sum().reset_index()
p_closed.columns = ['SK_ID_CURR','p_closed_sum(AMT_CREDIT)']

p_closed_cash = prev_closed[prev_closed.NAME_CONTRACT_TYPE.isin(['Cash loans','Consumer loans'])].groupby('SK_ID_CURR')['AMT_CREDIT'].sum().reset_index()
p_closed_cash.columns = ['SK_ID_CURR','p_closed_cash_sum(AMT_CREDIT)']

aggregations.append(p_closed)
aggregations.append(p_closed_cash)


payment_at_time = install.groupby(['SK_ID_CURR','DAYS_ENTRY_PAYMENT'])['NUM_INSTALMENT_VERSION'].count().reset_index()

def iagg(df, prefix):
    agg = df.groupby('SK_ID_CURR')['NUM_INSTALMENT_VERSION'].mean().reset_index()
    agg.columns = ['SK_ID_CURR', prefix+'mean(PAYMENT_AT_TIME)']
    return agg

aggregations.append(iagg(payment_at_time, 'ins_'))
aggregations.append(iagg(payment_at_time.query('DAYS_ENTRY_PAYMENT >= -365'), 'ins365_'))
aggregations.append(iagg(payment_at_time.query('DAYS_ENTRY_PAYMENT >= -720'), 'ins720_'))

install_last = install.drop_duplicates(subset=['SK_ID_PREV'], keep='first')
install_last.sort_values(by=['SK_ID_CURR','DAYS_INSTALMENT'],ascending=False,inplace=True)
install_last['interval'] = install_last.groupby('SK_ID_CURR')['DAYS_INSTALMENT'].shift(1) - install_last['DAYS_INSTALMENT']
i_agg = install_last.groupby('SK_ID_CURR').agg({
    'interval':['mean','max','min','std']
})
i_agg.columns = make_agg_names('ins_', i_agg)
i_agg.reset_index(inplace=True)

aggregations.append(i_agg)


bureau['AMT_CREDIT_DEBT_PERC_LIMIT'] = bureau['AMT_CREDIT_SUM_DEBT'].replace(0, np.nan) / bureau['AMT_CREDIT_SUM_LIMIT'].replace(0, np.nan)
bureau['AMT_CREDIT_DEBT_DIFF_LIMIT'] = bureau['AMT_CREDIT_SUM_DEBT'].replace(0, np.nan) - bureau['AMT_CREDIT_SUM_LIMIT'].replace(0, np.nan)

agg = {
    'AMT_CREDIT_DEBT_PERC_LIMIT':['mean','max','min'],
    'AMT_CREDIT_DEBT_DIFF_LIMIT':['mean','max','min','sum']
}
def cagg(df, agg, prefix):
    c_agg = df.groupby('SK_ID_CURR').agg(agg)
    c_agg.columns = make_agg_names(prefix, c_agg)
    c_agg.reset_index(inplace=True)
    return c_agg

aggregations.append(cagg(bureau, agg, 'b_'))
aggregations.append(cagg(bureau.query('CREDIT_ACTIVE == "Active"'), agg, 'b_active_'))
aggregations.append(cagg(bureau.query('CREDIT_ACTIVE == "Closed"'), agg, 'b_closed_'))

# 最小支払額と、実際に支払った額の差・比
credit['AMT_PAYMENT_TO_MIN_REGULARITY_RATIO'] = credit['AMT_PAYMENT_CURRENT'].fillna(0) / credit['AMT_INST_MIN_REGULARITY'].replace(0, np.nan)
credit['AMT_PAYMENT_TO_MIN_REGULARITY_DIFF'] = credit['AMT_PAYMENT_CURRENT'].fillna(0) - credit['AMT_INST_MIN_REGULARITY'].replace(0, np.nan)
credit['AMT_PAYMENT_TO_MIN_REGULARITY_SHORTAGE'] = credit['AMT_PAYMENT_TO_MIN_REGULARITY_DIFF'] * (credit['AMT_PAYMENT_TO_MIN_REGULARITY_DIFF'] < 0).astype(np.int32)

credit['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan, inplace=True)
credit['AMT_BALANCE_PER_LMT'] = credit['AMT_BALANCE'].replace(0, np.nan) / credit['AMT_CREDIT_LIMIT_ACTUAL']


agg = {
    'AMT_PAYMENT_TO_MIN_REGULARITY_RATIO': ['mean','min','max'],
    'AMT_PAYMENT_TO_MIN_REGULARITY_DIFF': ['mean','min','max'],
    'AMT_PAYMENT_TO_MIN_REGULARITY_SHORTAGE': ['mean','min','max'],
    'AMT_BALANCE_PER_LMT': ['std'],
    'CNT_DRAWINGS_ATM_CURRENT': ['max', 'min', 'std', 'sum'],
    'CNT_DRAWINGS_CURRENT': ['max', 'min', 'std', 'sum'],
    'AMT_DRAWINGS_ATM_CURRENT': ['std','min','max'],
    'AMT_DRAWINGS_CURRENT': ['std','min','max'],
    'AMT_INST_MIN_REGULARITY': ['std','min','max'],
    'AMT_PAYMENT_CURRENT': ['std','min','max'],
    'AMT_CREDIT_LIMIT_ACTUAL': ['std','min','max']
}

def cagg(df, agg, prefix):
    c_agg = df.groupby('SK_ID_CURR').agg(agg)
    c_agg.columns = make_agg_names(prefix, c_agg)
    c_agg.reset_index(inplace=True)
    return c_agg

aggregations.append(cagg(credit, agg, 'credit_'))
aggregations.append(cagg(credit.query('MONTHS_BALANCE >= -12'), agg, 'credit12_'))
aggregations.append(cagg(credit.query('MONTHS_BALANCE >= -6'), agg, 'credit6_'))


# save
df_features = pd.DataFrame()
df_features['SK_ID_CURR'] = df.SK_ID_CURR.unique()

df_features.shape

def merge(df_features, df):
    df_features = pd.merge(df_features, df, on='SK_ID_CURR', how='left')
    print('merged. shape: {}'.format(df_features.shape))
    return df_features


for a in aggregations:
    assert len(a) > 0

    df_features = merge(df_features, a)

df_features.to_feather('../model/add0805.f')