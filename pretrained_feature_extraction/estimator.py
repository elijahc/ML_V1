from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import explained_variance_score as fev
from keras.wrappers.scikit_learn import KerasRegressor

from deeporacle2 import DeepOracle

class SKDeepOracleRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, all_activity=None, optimizer='adam', loss='mse', epochs=15, batch_size=64):
        self.optimizer = optimizer
        self.loss = loss
        self.epochs=epochs
        self.batch_size=batch_size

        self.target_scale = (956,28,28,256)

        self.model = DeepOracle(target_shape=(28,28,256*2))
        self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss)

    def fit(self, X, y, **kwargs):

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, **kwargs)

        return self

    def predict(self, X, **kwargs):

        return self.model.predict(X)

    def score(self, X, y, **kwargs):
        y_pred = self.predict(X, **kwargs)
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
