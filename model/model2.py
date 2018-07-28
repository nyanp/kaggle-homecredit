import model


class Model2(model.Model):
    def __init__(self):
        super().__init__(name='model2',
                         remove_columns=[
                             'FLAG_DOCUMENT_5',
                             'FLAG_DOCUMENT_6',
                             'FLAG_DOCUMENT_8',
                             'FLAG_DOCUMENT_16',
                             'FLAG_DOCUMENT_18'
                         ])


if __name__ == "__main__":
    m = Model2()

    m.cv()
