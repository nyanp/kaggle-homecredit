import model

# Model8 + version3特徴(完了)
# CV 0.7939, LB 0.799
class Model9(model.Model):
    def __init__(self):
        super().__init__(name='model9',
                         add_columns=[
                             'PREDICTED_DPD',
                             'ins_min(DPD_with_prev)',
                             'ins_min(DPD_without_prev)',
                             'ins_min(DPD_RATIO)',
                             'ins_mean(DPD_RATIO)',
                             'ins_min(AMT_CREDIT_DEBT_PERC_NZ)',
                             'ins_max(AMT_CREDIT_DEBT_PERC_NZ)',
                             'ins_sum(AMT_CREDIT_DEBT_DIFF_NZ)',
                             'POS_PREDICTED',

                             # version-3特徴量

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

                             #'DPD_PREDICTED_REG_ALL'
                         ],
                         remove_columns=[
                             'FLAG_DOCUMENT_5',
                             'FLAG_DOCUMENT_6',
                             'FLAG_DOCUMENT_8',
                             'FLAG_DOCUMENT_16',
                             'FLAG_DOCUMENT_18'
                         ],
                         lgb_seed = 2,
                         drop_xna = True)


if __name__ == "__main__":
    m = Model9()

    m.cv()
