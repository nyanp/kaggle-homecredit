import model


class Model5(model.Model):
    def __init__(self):
        super().__init__(name='model5',
                         add_columns=[
                             'PREDICTED_DPD',
                             'ins_min(DPD_with_prev)',
                             'ins_min(DPD_without_prev)',
                             'ins_min(DPD_RATIO)',
                             'ins_mean(DPD_RATIO)',
                             'ins_min(AMT_CREDIT_DEBT_PERC_NZ)',
                             'ins_max(AMT_CREDIT_DEBT_PERC_NZ)',
                             'POS_PREDICTED',
                             'ins_sum(AMT_CREDIT_DEBT_DIFF_NZ)',
                             'ins_sum(AMT_CREDIT_DEBT_DIFF)'

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
    m = Model5()

    m.cv()
