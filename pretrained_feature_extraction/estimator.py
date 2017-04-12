import numpy as np
from keras.preprocessing import image as ki
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import explained_variance_score as fev
from keras.wrappers.scikit_learn import KerasRegressor
import inspect

from deeporacle2 import DeepOracle

class SKDeepOracleRegressor(BaseEstimator, RegressorMixin):

    def __init__(self,
            # Model params
            lname='block5_conv1',
            fc_size=300,
            dropout=0.5,
            # Fit Params
            epochs=20,
            batch_size=32
            ):
        self.lname = lname
        self.fc_size = fc_size
        self.dropout = dropout
        self.epochs=epochs
        self.batch_size=batch_size

        print('pre-loading images...')
        images = [ ki.img_to_array(ki.load_img('../data/images/%g.jpg'%id, target_size=(224,224))) for id in np.arange(956) ]
        images = np.array( images )
        self.images = images


    def fit(self, X, y):

        # Check params here if necessary

        train_idxs = X[:,0]
        train_images=self.images[train_idxs]

        self.model_ = DeepOracle(lname=self.lname, fc_size=self.fc_size, dropout=self.dropout)
        self.model_.fit(train_images, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

        return self

    def predict(self, X, **kwargs):

        test_idxs = X[:,0]
        test_images=self.images[test_idxs]
        return self.model_.predict(test_images, batch_size=self.batch_size)

    def score(self, X, y, **kwargs):
        y_pred = self.predict(X, **kwargs)
        import pdb; pdb.set_trace()
        return fev(y, y_pred, multioutput='uniform_average')

class KSDeepOracleRegressor(KerasRegressor):

    def call(self, layer_name='block4_conv4', optimizer='adam', loss='mse'):
        return DeepOracle(lname=layer_name).compile(optimizer=optimizer, loss=loss)

    def score(self, X, y, **kwargs):
        y_pred = self.predict(X, **kwargs)
        return fev(y, y_pred, multioutput='uniform_average')

if __name__ == '__main__':

    dor = KSDeepOracleRegressor()
    dor.fit([1,2,3],[4,5,6], epochs=20)
