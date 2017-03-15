from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import explained_variance_score as fev

from deeporacle import DeepOracle, build

class DeepOracleRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, all_activity=None, optimizer='adam', loss='mse', layer_1='block5_conv2', layer_2='block5_conv3', layer_3='block5_conv4', nb_epoch=10, batch_size=32):
        self.optimizer = optimizer
        self.loss = loss
        self.nb_epoch=nb_epoch
        self.batch_size=batch_size
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.layer_3 = layer_3
        self.layers = None

        self.target_scale = (956,56,56,128)
        self.all_activity=all_activity

        self.model = DeepOracle(target_shape=(56,56,128*3))
        self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss)

    def fit(self, X, y, **kwargs):
        new_layers = [self.layer_1, self.layer_2, self.layer_3]
        if self.layers != new_layers:
            self.layers = new_layers
            self.activations = build(self.layers, self.target_scale)

        self.model.fit(self.activations[X], self.all_activity[y], nb_epoch=round(self.nb_epoch), **kwargs)

        return self

    def predict(self, X, **kwargs):

        return self.model.predict(self.activations[X])

    def score(self, X, y, **kwargs):
        y_pred = self.predict(X, **kwargs)
        return fev(self.all_activity[y], y_pred, multioutput='uniform_average')

if __name__ == '__main__':

    dor = DeepOracleRegressor()
    dor.fit([1,2,3],[4,5,6], nb_epoch=20)
