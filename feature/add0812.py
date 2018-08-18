import pandas as pd
import numpy as np

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


def make_features(df, prev, bureau, bb, pos, credit, install):
    df_features = pd.DataFrame()
    df_features['SK_ID_CURR'] = df.SK_ID_CURR.unique()

    # 同額の申し込み
    def same_credit_within(df, prev, days):
        df_ = df[['SK_ID_CURR','AMT_GOODS_PRICE','AMT_CREDIT']]
        df_.columns = ['SK_ID_CURR','AMT_GOODS_PRICE_CURR','AMT_CREDIT_CURR']
        prev_ = pd.merge(prev[prev.DAYS_DECISION >= -days], df_, on='SK_ID_CURR', how='left')

        prev_['N_SAME_GOODS'] = prev_['AMT_GOODS_PRICE'] == prev_['AMT_GOODS_PRICE_CURR']
        prev_['N_SAME_CREDIT'] = prev_['AMT_CREDIT'] == prev_['AMT_CREDIT']

        agg = aggregate(prev_, {
            'N_SAME_GOODS':['sum'],
            'N_SAME_CREDIT':['sum']
        }, 'p{}_'.format(days))
        return agg

    agg365 = same_credit_within(df, prev, 365)
    agg180 = same_credit_within(df, prev, 180)

    df_features = merge(df_features, agg365)
    df_features = merge(df_features, agg180)

    install['DAYS_INSTALMENT_INTERVAL'] = install['DAYS_INSTALMENT'] - install.groupby('SK_ID_PREV')[
        'DAYS_INSTALMENT'].shift(1)

    agg = aggregate(install, {'DAYS_INSTALMENT_INTERVAL': ['max', 'mean', 'std']}, 'ins_')

    df_features = merge(df_features, agg)

    col_house = [
        'APARTMENTS_AVG', 'APARTMENTS_MEDI',
        'APARTMENTS_MODE', 'BASEMENTAREA_AVG',
        'BASEMENTAREA_MEDI', 'BASEMENTAREA_MODE',
        'COMMONAREA_AVG', 'COMMONAREA_MEDI', 'COMMONAREA_MODE',
        'ELEVATORS_AVG', 'ELEVATORS_MEDI', 'ELEVATORS_MODE',
        'EMERGENCYSTATE_MODE',
        'ENTRANCES_AVG', 'ENTRANCES_MEDI', 'ENTRANCES_MODE',
        'FLOORSMAX_AVG', 'FLOORSMAX_MEDI', 'FLOORSMAX_MODE',
        'FLOORSMIN_AVG', 'FLOORSMIN_MEDI', 'FLOORSMIN_MODE',
        'FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE',
        'LANDAREA_AVG', 'LANDAREA_MEDI', 'LANDAREA_MODE',
        'LIVE_CITY_NOT_WORK_CITY', 'LIVE_REGION_NOT_WORK_REGION',
        'LIVINGAPARTMENTS_AVG', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAPARTMENTS_MODE',
        'LIVINGAREA_AVG', 'LIVINGAREA_MEDI', 'LIVINGAREA_MODE',
        'NAME_HOUSING_TYPE',
        'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_MODE',
        'NONLIVINGAREA_AVG', 'NONLIVINGAREA_MEDI', 'NONLIVINGAREA_MODE',
        'REGION_POPULATION_RELATIVE', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
        'TOTALAREA_MODE',
        'WALLSMATERIAL_MODE',
        'YEARS_BUILD_AVG', 'YEARS_BUILD_MEDI', 'YEARS_BUILD_MODE']

    col_user = [
        'CNT_CHILDREN',
        'CNT_FAM_MEMBERS',
        'CODE_GENDER',
        'NAME_EDUCATION_TYPE'
    ]

    d_fam = df.groupby(col_house)['AMT_CREDIT'].count().reset_index().rename(columns={'AMT_CREDIT': 'N_FAMILIES'})
    d_fam = d_fam.query('N_FAMILIES > 1')
    d_same = df.groupby(col_house + col_user).agg(
        {
            'AMT_CREDIT': ['count'],
            'DAYS_BIRTH': ['min', 'max'],
            'EXT_SOURCE_1': ['mean'],
            'EXT_SOURCE_2': ['mean'],
            'EXT_SOURCE_3': ['mean']
        }
    ).reset_index()
    d_same.columns = col_house + col_user + ['N_USER', 'min(DAYS_BIRTH)', 'max(DAYS_BIRTH)', 'mean(EXT1)', 'mean(EXT2)',
                                             'mean(EXT3)']
    d_same['APPLICATION_DIFF'] = d_same['max(DAYS_BIRTH)'] - d_same['min(DAYS_BIRTH)']
    d_same = d_same.query('N_USER > 1 and APPLICATION_DIFF < 365')

    df_ = pd.merge(df, d_fam, on=col_house, how='left')
    df_ = pd.merge(df_, d_same, on=col_house + col_user, how='left')
    df_['DAYS_FROM_LAST_DUP'] = df_['min(DAYS_BIRTH)'] - df_['DAYS_BIRTH']

    df_features = merge(df_features, df_[
        ['SK_ID_CURR', 'N_FAMILIES', 'DAYS_FROM_LAST_DUP', 'N_USER', 'mean(EXT1)', 'mean(EXT2)', 'mean(EXT3)']])

    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['ENDDATE_DIFF'] = prev['DAYS_LAST_DUE'] - prev['DAYS_LAST_DUE_1ST_VERSION']
    prev['DAYS_LAST_DUE_FUTURE'] = prev['DAYS_LAST_DUE_1ST_VERSION'] * (prev['DAYS_LAST_DUE_1ST_VERSION'] > 0).astype(
        np.int32)
    prev['DAYS_LAST_DUE_PAST'] = prev['DAYS_LAST_DUE_1ST_VERSION'] * (prev['DAYS_LAST_DUE_1ST_VERSION'] < 0).astype(
        np.int32)

    agg = {
        'DAYS_LAST_DUE_FUTURE': ['mean'],
        'DAYS_LAST_DUE_PAST': ['mean'],
        'ENDDATE_DIFF': ['mean']
    }

    pagg_cash = aggregate(prev.query('NAME_CONTRACT_TYPE == "Cash loans"'), agg, 'p_cash_')
    pagg_cons = aggregate(prev.query('NAME_CONTRACT_TYPE == "Consumer loans"'), agg, 'p_consumer_')
    pagg_credit = aggregate(prev.query('NAME_CONTRACT_TYPE == "Revloving loans"'), agg, 'p_credit_')

    pagg_cash720 = aggregate(prev.query('NAME_CONTRACT_TYPE == "Cash loans" and DAYS_DECISION >= -720'), agg,
                             'p_cash720_')
    pagg_cons720 = aggregate(prev.query('NAME_CONTRACT_TYPE == "Consumer loans" and DAYS_DECISION >= -720'), agg,
                             'p_consumer720_')
    pagg_credit720 = aggregate(prev.query('NAME_CONTRACT_TYPE == "Revloving loans" and DAYS_DECISION >= -720'), agg,
                               'p_credit720_')

    df_features = merge(df_features, pagg_cash)
    df_features = merge(df_features, pagg_cons)
    df_features = merge(df_features, pagg_credit)
    df_features = merge(df_features, pagg_cash720)
    df_features = merge(df_features, pagg_cons720)
    df_features = merge(df_features, pagg_credit720)

    prev_completed = prev[prev.SK_ID_PREV.isin(completed(pos, credit))]

    def completed_amt(prev, days, prefix):
        p = prev[prev.DAYS_DECISION >= -days]
        agg = p.groupby('SK_ID_CURR')['AMT_CREDIT'].sum().reset_index()
        agg.columns = ['SK_ID_CURR', prefix + 'sum(AMT_CREDIT)']
        return agg

    p1 = completed_amt(prev_completed, 720, 'p_closed720_')
    p2 = completed_amt(prev_completed, 365, 'p_closed365_')

    def completed_bureau(bureau, days, prefix):
        b_closed = bureau.query('CREDIT_ACTIVE == "Closed"')
        b_closed = b_closed[b_closed.DAYS_CREDIT >= -days]
        agg = b_closed.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum().reset_index()
        agg.columns = ['SK_ID_CURR', prefix + 'sum(AMT_CREDIT_SUM)']
        return agg

    b1 = completed_bureau(bureau, 720, 'b_closed720_')
    b2 = completed_bureau(bureau, 365, 'b_closed365_')

    df_features = merge(df_features, p1)
    df_features = merge(df_features, p2)
    df_features = merge(df_features, b1)
    df_features = merge(df_features, b2)

    def future_payment(days, df, bureau, pos):
        # bureau:
        b = bureau.query('DAYS_CREDIT_UPDATE >= -365 and DAYS_CREDIT_ENDDATE > 0').copy()
        b['CNT_INSTALMENT_FUTURE'] = b['DAYS_CREDIT_ENDDATE'].clip(np.nan, days) / 30
        b['FUTURE_PAYMENT'] = b['AMT_ANNUITY'] * b['CNT_INSTALMENT_FUTURE']

        # pos-cash:
        p = pos.drop_duplicates(subset=['SK_ID_PREV'], keep='first').copy()
        p = pd.merge(prev, p[['SK_ID_PREV', 'CNT_INSTALMENT_FUTURE']], on='SK_ID_PREV', how='left')
        p = p[p.CNT_INSTALMENT_FUTURE > 0]
        p['CNT_INSTALMENT_FUTURE'] = p['CNT_INSTALMENT_FUTURE'].clip(np.nan, days / 30)
        p['FUTURE_PAYMENT'] = p['AMT_ANNUITY'] * p['CNT_INSTALMENT_FUTURE']

        print(b.shape)
        print(p.shape)

        agg = {
            'CNT_INSTALMENT_FUTURE': ['sum'],
            'FUTURE_PAYMENT': ['sum'],
            'AMT_ANNUITY': ['sum']
        }

        prefix_b = 'b_future{}_'.format(days)
        prefix_p = 'p_future{}_'.format(days)

        bagg = aggregate(b, agg, prefix_b)
        pagg = aggregate(p, agg, prefix_p)

        df_ = pd.merge(df, bagg, on='SK_ID_CURR', how='left')
        df_ = pd.merge(df_, pagg, on='SK_ID_CURR', how='left')

        df_['CNT_INSTALMENT_FUTURE'] = df_['AMT_CREDIT'] / df_['AMT_ANNUITY']
        df_['CNT_INSTALMENT_FUTURE'] = df_['CNT_INSTALMENT_FUTURE'].clip(np.nan, days / 30)
        df_['FUTURE_PAYMENT'] = df_['CNT_INSTALMENT_FUTURE'] * df_['AMT_ANNUITY']
        df_['FUTURE_INCOME'] = df_['CNT_INSTALMENT_FUTURE'] * df_['AMT_INCOME_TOTAL']
        df_['TTL_PAYMENT_{}'.format(days)] = df_['FUTURE_PAYMENT'] + df_[prefix_b + 'sum(FUTURE_PAYMENT)'].fillna(0) + \
                                             df_[prefix_p + 'sum(FUTURE_PAYMENT)'].fillna(0)
        df_['TTL_BALANCE_{}'.format(days)] = df_['FUTURE_INCOME'] - df_['TTL_PAYMENT_{}'.format(days)]
        df_['PAY_PERC_{}'.format(days)] = df_['FUTURE_PAYMENT'] / (
                    df_[prefix_b + 'sum(FUTURE_PAYMENT)'].fillna(0) + df_[prefix_p + 'sum(FUTURE_PAYMENT)'].fillna(0))

        return df_[
            ['SK_ID_CURR', 'TTL_PAYMENT_{}'.format(days), 'TTL_BALANCE_{}'.format(days), 'PAY_PERC_{}'.format(days),
             prefix_b + 'sum(FUTURE_PAYMENT)', prefix_p + 'sum(FUTURE_PAYMENT)']]

    f180 = future_payment(180, df, bureau, pos)
    f365 = future_payment(365, df, bureau, pos)

    df_features = merge(df_features, f180)
    df_features = merge(df_features, f365)

    pagg = prev_completed.groupby('SK_ID_CURR')['AMT_CREDIT'].sum().reset_index()
    pagg.columns = ['SK_ID_CURR', 'p_completed_sum(AMT_CREDIT)']

    df_features = merge(df_features, pagg)

    bb_ = pd.merge(bb, bureau[['SK_ID_BUREAU', 'SK_ID_CURR', 'CREDIT_TYPE']], on='SK_ID_BUREAU', how='left')

    agg = {
        'MONTHS_BALANCE': ['count']
    }

    b1 = aggregate(bb_, agg, 'b_')
    b2 = aggregate(bb_.query('MONTHS_BALANCE >= -12'), agg, 'b12_')
    b3 = aggregate(bb_.query('MONTHS_BALANCE >= -12 and CREDIT_TYPE == "Consumer credit"'), agg, 'b_consumer12_')

    df_features = merge(df_features, b1)
    df_features = merge(df_features, b2)
    df_features = merge(df_features, b3)

    install_ = install.copy()
    install_['SEASON'] = (install_['DAYS_INSTALMENT'] // 365).astype(np.int32)
    install_.head()

    for i in range(3, 8):
        i_ = install_[install_.SEASON == -i]
        iagg = i_.groupby('SK_ID_CURR')['AMT_PAYMENT'].agg(['sum', 'count']).reset_index()
        iagg.columns = ['SK_ID_CURR', 'ins_{}y_sum(AMT_PAYMENT)'.format(i), 'ins_{}y_count(AMT_PAYMENT)'.format(i)]
        df_features = merge(df_features, iagg)

    active_ids = active(pos, credit)

    prev_active = prev[prev.SK_ID_PREV.isin(active_ids)]

    agg = {
        'CNT_PAYMENT': ['mean']
    }

    p1 = aggregate(prev_active, agg, 'p_active_')
    df_features = merge(df_features, p1)

    selected_features = [
        "p_consumer720_mean(ENDDATE_DIFF)",
        #"mean(EXT3)",
        "ins_mean(DAYS_INSTALMENT_INTERVAL)",
        "p_consumer_mean(DAYS_LAST_DUE_PAST)",
        "p_cash_mean(DAYS_LAST_DUE_PAST)",
        #"mean(EXT2)",
        #"N_USER",
        "p_consumer720_mean(DAYS_LAST_DUE_PAST)",
        "p_cash720_mean(ENDDATE_DIFF)",
        "p_cash720_mean(DAYS_LAST_DUE_PAST)",
        "p365_sum(N_SAME_GOODS)",
        "p180_sum(N_SAME_CREDIT)",
        "b_closed720_sum(AMT_CREDIT_SUM)",
        "p_consumer_mean(ENDDATE_DIFF)",
        "p_closed365_sum(AMT_CREDIT)",
        "p_closed720_sum(AMT_CREDIT)",
        "b_closed365_sum(AMT_CREDIT_SUM)",
        "ins_std(DAYS_INSTALMENT_INTERVAL)",
        #"DAYS_FROM_LAST_DUP",
        "N_FAMILIES",
        "p_cash720_mean(DAYS_LAST_DUE_FUTURE)",
        "p_consumer720_mean(DAYS_LAST_DUE_FUTURE)",
        "p_future365_sum(FUTURE_PAYMENT)",
        "b_count(MONTHS_BALANCE)",
        "p_future180_sum(FUTURE_PAYMENT)",
        "PAY_PERC_180"
    ]
    return df_features[['SK_ID_CURR']+selected_features]
