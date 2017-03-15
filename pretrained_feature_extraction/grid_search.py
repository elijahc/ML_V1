import numpy as np
import scipy.io as sio
from vgg19 import VGG19
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from estimator import DeepOracleRegressor
from deeporacle import train_test, build_random

import scipy.stats as stats

if __name__ == '__main__':
    mat_file = '../data/02mean_d1.mat'
    activity_file = '../data/02_stats.mat'
    print('loading mat data...', mat_file)
    mat_contents = sio.loadmat(mat_file)
    # activity_contents = sio.loadmat(activity_file)
    activity = mat_contents['activity']
    # sem_activity = activity_contents['resp_sem'].swapaxes(0,1)

    # Small Natural Images and gratings
    idxs = np.arange(540)[::2]
    idxs = np.concatenate([idxs, np.arange(540,732)])

    idxs_train, idxs_test = train_test(idxs, 0.8)
    block3 = ['block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4']
    block4 = ['block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4']
    block5 = ['block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4']
    all_blocks = []
    all_blocks.extend(block3)
    all_blocks.extend(block4)
    all_blocks.extend(block5)

    #layers, activations = build_random(using=block5, choose=3, target_scale=(956,56,56,128))

    #X_train = activations[idxs]
    #y_train = activity[idxs]

    dor = DeepOracleRegressor(all_activity=activity)
    param_grid = {'batch_size': [8,16,32,64,128]}
    layer_param_grid = {
            'layer_1': all_blocks,
            'layer_2': all_blocks,
            'layer_3': all_blocks,
            'all_activity':[activity]}
    clf = RandomizedSearchCV(dor, layer_param_grid, fit_params={'batch_size':64}, verbose=1, cv=5, n_jobs=1, n_iter=5)
    clf.fit(idxs_train, idxs_train)

    print('Best Parameter set found on development set:')
    print()
    print(clf.best_params_)
    print()
    print('Grid scores on development set:')
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    import pdb; pdb.set_trace()
    for split, batch_size in zip(['split'+str(n)+'_test_score' for n in np.arange(5)], [8,16,32,64,128]):
        for val in clf.cv_results_[split]:
            print("%g, %0.03f" % (batch_size, val))


    print()

    import pdb; pdb.set_trace()
