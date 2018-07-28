import model


class Model3(model.Model):
    def __init__(self):
        super().__init__(name='model3_seed2',
                         add_columns=[
                             'PREDICTED_DPD',
                             'ins_min(DPD_with_prev)',
                             'ins_min(DPD_without_prev)',
                             'ins_min(DPD_RATIO)',
                             'ins_mean(DPD_RATIO)'
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
    m = Model3()

    m.cv()
