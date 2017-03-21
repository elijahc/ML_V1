from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import explained_variance_score as fev

from deeporacle import DeepOracle, build

class DeepOracleRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, all_activity=None, optimizer='adam', loss='mse', epochs=15, batch_size=64):
        self.optimizer = optimizer
        self.loss = loss
        self.epochs=epochs
        self.batch_size=batch_size

        self.target_scale = (956,14,14,512)

        self.model = DeepOracle(target_shape=(14,14,512*3))
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

if __name__ == '__main__':

    dor = DeepOracleRegressor()
    dor.fit([1,2,3],[4,5,6], epochs=20)
