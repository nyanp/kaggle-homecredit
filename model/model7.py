import model

# Model6 + version3特徴(途中)
class Model7(model.Model):
    def __init__(self):
        super().__init__(name='model7',
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

                             # CNT_INSTALMENT_AHEAD:前倒しの月数
                             'pos_min(CNT_INSTALMENT_AHEAD)',
                             'pos_max(CNT_INSTALMENT_AHEAD)',

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
                             'credit12_max(AMT_BALANCE_PER_LMT)' # 上と同様
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
    m = Model7()

    m.cv()
