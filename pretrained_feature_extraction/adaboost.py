import scipy.io as sio
from scipy.sparse import csr_matrix
import numpy as np
from keras.preprocessing import image as ki
from estimator import SKDeepOracleRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from deeporacle2 import DeepOracle, train_test

late_file = '../data/02_stats_late.mat'
print('loading ', late_file, '...')

l_activity_contents = sio.loadmat(late_file)

l_activity = l_activity_contents['resp_mean'].swapaxes(0,1)

# Small Images
idxs = np.arange(540)[::2]

train_idxs, valid_idxs = train_test(idxs,0.8)

images = [ ki.img_to_array(ki.load_img('../data/images/%g.jpg'%id, target_size=(224,224))) for id in np.arange(956) ]
images = np.array( images )

# X = train_idxs.reshape(-1,1)
X = idxs.reshape(-1,1)
# X = images[train_idxs]

# y = l_activity[train_idxs][:,8]
y = l_activity[idxs][:,8]

VX = images[valid_idxs]
Vy = l_activity[valid_idxs]

dor = SKDeepOracleRegressor()

abr = AdaBoostRegressor(base_estimator=dor, n_estimators=20)

scores = cross_val_score(abr, X,y)
import pdb; pdb.set_trace()
