import model


class Model4(model.Model):
    def __init__(self):
        param = {
            'objective': 'binary',
            'num_leaves': 32,
            'learning_rate': 0.04,
            'colsample_bytree': 0.95,
            'subsample': 0.872,
            'max_depth': -1, # changed
            'reg_alpha': 0.04,
            'reg_lambda': 0.073,
            'min_split_gain': 0.0222415,
            'min_child_weight': 40,
            'metric': 'auc',
            'n_estimators': 10000
        }

        super().__init__(name='model4',
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
                         drop_xna = True,
                         param = param)


if __name__ == "__main__":
    m = Model4()

    m.cv()
