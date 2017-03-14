from sklearn.base import BaseEstimator
from sklearn.metrics import explained_variance_score as fev

from deeporacle import DeepOracle

class DeepOracleRegressor(BaseEstimator):

    def __init__(self, optimizer='rmsprop', loss='mse'):
        self.optimizer = optimizer
        self.loss = loss

        self.model = DeepOracle()
        self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

        return self

    def predict(self, X, **kwargs):
        return self.model.predict(X)

    def score(self, X, y, **kwargs):
        y_pred = self.predict(X, **kwargs)
        return fev(y, y_pred, multioutput='raw_values')

if __name__ == '__main__':

    dor = DeepOracleRegressor()
    dor.fit([1,2,3],[4,5,6], nb_epoch=20)
