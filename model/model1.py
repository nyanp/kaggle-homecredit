import model

class Model1(model.Model):
    def __init__(self):
        #super().__init__(name='model1', drop_xna=True, remove_columns=['FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8']) # 0.791908
        super().__init__(name='model1', drop_xna=True)

if __name__ == "__main__":
    m = Model1()

    m.cv()
