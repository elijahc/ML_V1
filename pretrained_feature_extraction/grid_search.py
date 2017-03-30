import numpy as np
import gc
import scipy.io as sio
from vgg19 import VGG19
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from estimator import DeepOracleRegressor
from deeporacle import train_test, build
import itertools
import csv

import scipy.stats as stats

def train(layers):
    target_scale = (956,28,28,256)
    activations = build(layers, target_scale)

    X_train = activations[idxs]
    y_train = activity[idxs]

    dor = DeepOracleRegressor(all_activity=activity)

    param_grid = {'batch_size': [64]}

    clf = GridSearchCV(dor, param_grid, verbose=1, cv=5, n_jobs=1)
    clf.fit(X_train, y_train)
    activations=None
    del activations, X_train, y_train
    gc.collect()

    return clf

def print_out(clf, layers):
    print('Best Parameter set found on development set:')
    print()
    print(clf.best_params_)
    print()
    print('Grid scores on development set:')
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    print()
    print('writing to csv...')
    with open('all_blocks_early.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)

        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, list(layers)))
            writer.writerow([layers, params, mean, std*2])

if __name__ == '__main__':
    mat_file = '../data/02mean_d1.mat'
    activity_file = '../data/02_stats_early.mat'
    print('loading mat data...', mat_file)
    mat_contents = sio.loadmat(mat_file)
    activity_contents = sio.loadmat(activity_file)
    activity = activity_contents['resp_mean'].swapaxes(0,1)
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

    combos = [e for e in itertools.combinations(all_blocks, 2)]

    for layers in combos:

        clf = train(list(layers))
        print_out(clf, layers)

    # clf = RandomizedSearchCV(dor, layer_param_grid, fit_params={'batch_size':32}, verbose=1, cv=5, n_jobs=1, n_iter=20)
