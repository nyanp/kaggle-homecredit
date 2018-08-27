import model_base
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from keras.models import load_model

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class NN(model_base.ModelBase):
    def __init__(self, name, comment='', basepath='../feature/features_all_std.f', units=None, bns=None, dropouts=None, lr=0.01, batch_size=512, epochs=30):
        super().__init__(name, comment)

        df = pd.read_feather(basepath)
        self.x = df.drop('TARGET', axis=1).fillna(0)  # TODO
        self.y = df['TARGET']

        print('X:{}'.format(self.x.shape))

        training = self.y.notnull()
        testing = self.y.isnull()
        self.X_train = self.x[training].reset_index(drop=True)
        self.X_test = self.x[testing].reset_index(drop=True)
        self.y_train = self.y[training].reset_index(drop=True)

        if units is None:
            self.units = [800, 320, 64, 52, 24]
        else:
            self.units = units

        if bns is None:
            self.bns = [False, True, True, True, True]
        else:
            self.bns = bns

        if dropouts is None:
            self.dropouts = [0.6,0.3,0.3,0.3,0.3]
        else:
            self.dropouts = dropouts

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

    def get_train(self):
        return self.X_train, self.y_train

    def get_test(self):
        return self.X_test

    def on_start_cv(self):
        self.log('units: {}'.format(self.units))
        self.log('bn: {}'.format(self.bns))
        self.log('batch-size: {}'.format(self.batch_size))
        self.log('learning-rate: {}'.format(self.lr))

    def make_model(self):
        nn = Sequential()

        for i, u in enumerate(self.units):
            nn.add(Dense(units=u, kernel_initializer='normal', input_dim=self.X_train.shape[1] if i == 0 else None))
            nn.add(PReLU())
            if self.bns[i]:
                nn.add(BatchNormalization())
            if self.dropouts[i] > 0:
                nn.add(Dropout(self.dropouts[i]))

        nn.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        adam = Adam(lr=self.lr)
        nn.compile(loss='binary_crossentropy', optimizer=adam)

        return nn

    def train(self, train_x, train_y, valid_x, valid_y) -> None:
        ES = EarlyStopping(patience=2)
        MC = ModelCheckpoint('weights.hdf5', save_best_only=True)

        self.nn = self.make_model()
        self.nn.fit(train_x,
               train_y,
               batch_size=self.batch_size,
               validation_data = (valid_x, valid_y),
               epochs=self.epochs,
               verbose=2,
               callbacks=[ES, MC, roc_callback(training_data=(train_x, train_y),validation_data=(valid_x, valid_y))])

        self.nn = load_model('weights.hdf5')

    def predict(self, test_x):
        return self.nn.predict(test_x).flatten()



if __name__ == "__main__":

    for i in range(10):
        name = 'nn_normal_v0_seed{}'.format(i)
        m = NN(name, epochs=15, batch_size=256)
        auc, oof, preds = m.cv(5, submission='../output/'+name+'.csv', save_oof='../stack/{}_'+name+'.npy', seed=i)

    for i in range(10):
        name = 'nn_normal_v0_seed{}_lr0003'.format(i)
        m = NN(name, epochs=45, batch_size=256, lr=0.003)
        auc, oof, preds = m.cv(5, submission='../output/'+name+'.csv', save_oof='../stack/{}_'+name+'.npy', seed=i)
