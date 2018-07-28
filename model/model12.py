import model

#
class Model12(model.Model):
    def __init__(self, name=None, param=None):
        if name is not None:
            model_name = name
        else:
            model_name = 'model12'

        super().__init__(name=model_name,
                         add_columns=[
                             'PREDICTED_X14Y-1',
                             'ins_min(DPD_with_prev)',
                             'ins_min(DPD_without_prev)',
                             'ins_min(DPD_RATIO)',
                             'ins_mean(DPD_RATIO)',
                             'ins_min(AMT_CREDIT_DEBT_PERC_NZ)',
                             'ins_max(AMT_CREDIT_DEBT_PERC_NZ)',
                             'ins_sum(AMT_CREDIT_DEBT_DIFF_NZ)',
                             'POS_PREDICTED',

                             # SEASON-3 期間限定＆前倒し返済関連

                             # CNT_INSTALMENT_AHEAD:前倒し月の割合
                             'pos_min(CNT_INSTALMENT_AHEAD_RATIO)',
                             'pos12_mean(CNT_INSTALMENT_AHEAD_RATIO)',
                             'pos12_max(CNT_INSTALMENT_AHEAD_RATIO)',

                             # CNT_INSTALMENT_AHEAD:前倒しの月数
                             'pos_min(CNT_INSTALMENT_AHEAD)',
                             'pos_max(CNT_INSTALMENT_AHEAD)',
                             'pos12_min(CNT_INSTALMENT_AHEAD)',
                             'pos12_max(CNT_INSTALMENT_AHEAD)',

                             # CREDIT_LIMIT_LAST_BY_MIN: 1つのリボ払いについて、限度額の最新の値と最小値の比率。途中で限度額を更新した場合に1でなくなる。
                             'credit_prev_mean(CREDIT_LIMIT_LAST_BY_MIN)',

                             # AMT_CREDIT_LIMIT_ACTUAL: クレジット限度額(0の行は削除)　TODO: mean以外のaggregationも効きそう
                             'credit_mean(AMT_CREDIT_LIMIT_ACTUAL)',
                             'credit6_mean(AMT_CREDIT_LIMIT_ACTUAL)',
                             'credit12_mean(AMT_CREDIT_LIMIT_ACTUAL)', # CV総合は上がっていないが、3/5Foldで改善、かつ他の期間で上がっているので採用

                             # AMT_BALANCE_PER_LMT: クレジット限度額に対する利用額の比率
                             'credit_max(AMT_BALANCE_PER_LMT)',
                             'credit_mean(AMT_BALANCE_PER_LMT)',
                             'credit6_min(AMT_BALANCE_PER_LMT)',
                             'credit6_max(AMT_BALANCE_PER_LMT)',
                             'credit12_mean(AMT_BALANCE_PER_LMT)',
                             'credit12_max(AMT_BALANCE_PER_LMT)', # 上と同様


                             'ins180_max(AMT_PAYMENT)',
                             'ins180_min(AMT_PAYMENT)',
                             'ins180_sum(AMT_PAYMENT)',
                             'ins180_std(AMT_PAYMENT)',
                             'ins180_mean(AMT_PAYMENT)',
                             'ins365_sum(AMT_PAYMENT)',

                             'ins180_mean(PAYMENT_PERC)',
                             'ins180_var(PAYMENT_PERC)',
                             'ins180_std(PAYMENT_PERC)',
                             'ins180_min(PAYMENT_PERC)',
                             'ins365_mean(PAYMENT_PERC)',
                             'ins365_min(PAYMENT_PERC)',
                             'ins365_var(PAYMENT_PERC)',
                             'ins365_std(PAYMENT_PERC)',

                             'ins180_sum(AMT_INSTALMENT)',
                             'ins180_mean(AMT_INSTALMENT)',
                             'ins365_sum(AMT_INSTALMENT)',
                             'ins365_std(AMT_INSTALMENT)',
                             'ins365_max(AMT_INSTALMENT)',

                             'ins180_max(PAYMENT_DIFF)',
                             'ins180_mean(PAYMENT_DIFF)',
                             'ins365_max(PAYMENT_DIFF)',
                             'ins365_var(PAYMENT_DIFF)',
                             'ins365_std(PAYMENT_DIFF)',
                             'ins365_mean(PAYMENT_DIFF)',

                             'ins180_max(DPD)',
                             'ins180_mean(DPD)',
                             'ins180_std(DPD)',
                             'ins180_sum(DPD)',
                             'ins365_max(DPD)',
                             'ins365_mean(DPD)',
                             'ins365_std(DPD)',
                             'ins365_sum(DPD)',

                             'ins180_sum(DBD)',
                             'ins180_max(DBD)',
                             'ins180_std(DBD)',
                             'ins180_mean(DBD)',
                             'ins365_mean(DBD)',

                             'ins365_mean(DAYS_ENTRY_PAYMENT_INTERVAL)',
                             'ins365_min(DAYS_ENTRY_PAYMENT_INTERVAL)',

                             'ins180_sum(DAYS_ENTRY_PAYMENT)',
                             'ins365_mean(DAYS_ENTRY_PAYMENT)',

                             # SEASON-4 applicationと他テーブルの複合
                             'PREV_TO_CURR_ANNUITY_RATIO_CASH',
                             'PREV_TO_CURR_CREDIT_RATIO_CASH',
                             'AMT_CREDIT_BY_ORG_CONTRACT_RATIO',
                             'PREV_TO_CURR_ANNUITY_RATIO',
                             'AMT_ANNUITY_BY_ORG_CONTRACT_DIFF',
                             'PREV_TO_CURR_ANNUITY_RATIO_REVOLVING',
                             'AMT_ANNUITY_BY_ORG_CONTRACT',
                             'PREV_TO_CURR_CREDIT_RATIO_REVOLVING',
                             'MEAN_PREV_REVOLVING_ANNUITY',
                             'MEAN_PREV_CREDIT',
                             'AMT_CREDIT_BY_ORG_CONTRACT_DIFF',
                             'AMT_ANNUITY_BY_ORG_CONTRACT_RATIO',
                             'MEAN_PREV_CASH_ANNUITY',
                             'MEAN_PREV_ANNUITY',
                             'MEAN_PREV_REVOLVING_CREDIT',
                             'AMT_GOODS_PRICE_BY_ORG_CONTRACT'

                         ],
                         remove_columns=[
                             'FLAG_DOCUMENT_5',
                             'FLAG_DOCUMENT_6',
                             'FLAG_DOCUMENT_8',
                             'FLAG_DOCUMENT_16',
                             'FLAG_DOCUMENT_18'
                         ],
                         lgb_seed = 2,
                         drop_xna = True,
                         param = param)


if __name__ == "__main__":
    param = {
        'objective': 'binary',
        'num_leaves': 32,
        'learning_rate': 0.04,
        'colsample_bytree': 0.95,
        'subsample': 0.872,
        'max_depth': 8,
        'reg_alpha': 0.04,
        'reg_lambda': 0.073,
        'min_split_gain': 0.0222415,
        'min_child_weight': 40,
        'metric': 'auc',
        'n_estimators': 10000
    }

    i = 0

    f = open('log_gridsearch.txt', 'w')

    for a in [0, 0.04, 0.1]:
        for b in [0, 0.073, 0.1]:
            for m in [20, 40, 80]:
                for c in [0.2, 0.5]:
                    param['reg_alpha'] = a
                    param['reg_lambda'] = b
                    param['min_child_weight'] = m
                    param['colsample_bytree'] = c

                    i = i + 1

                    if i == 1:
                        continue

                    name = 'cv5_m12_a{}_b{}_m{}_c{}'.format(a,b,m,c)

                    model = Model12(param=param, name=name)
                    _, auc = model.cv(nfolds=5)

                    f.write('{},{}\n'.format(name, auc))
