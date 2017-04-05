import scipy.io as sio
import numpy as np
from estimator import KSDeepOracleRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor
from deeporacle2 import DeepOracle, train_test

late_file = '../data/02_stats_late.mat'
print('loading ', late_file, '...')

l_activity_contents = sio.loadmat(late_file)

l_activity = l_activity_contents['resp_mean'].swapaxes(0,1)

# Small Images
idxs = np.arange(540)[::2]

train_idxs, test_idxs = train_test(idxs,0.8)

X = train_idxs.reshape(-1,1)
y = l_activity[train_idxs]

dor = KerasRegressor(build_fn=DeepOracle)
abr = AdaBoostRegressor(base_estimator=dor, n_estimators=10)

abr.fit(X,y)
