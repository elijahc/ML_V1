import scipy.io as sio
import joblib
import numpy as np
import h5py
import tensorflow as tf
from tqdm import tqdm
import cv2
from vgg19 import VGG19
from keras.preprocessing import image as ki
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from imagenet_utils import preprocess_input
from scipy.ndimage.interpolation import zoom
from selectivity import si
from sklearn.base import BaseEstimator
from sklearn.metrics import explained_variance_score as fev
import csv

def DeepOracle(lname='block5_conv1', fc_size=300, dropout=0.5):

    base = VGG19(weights='imagenet')
    layers = base.layers
    target_layer = base.get_layer(lname)
    target_idx = layers.index(target_layer) + 1
    layers = layers[:target_idx]
    for l in layers:
        l.trainable=False

    in_shape = [ int(e) for e in list(target_layer.output.shape[1:]) ]
    in_shape = tuple(in_shape)

    # Convolutional readout network
    readout = [
    Convolution2D(2, (1, 1), activation='relu', padding='same',name='readout_conv1', input_shape=in_shape),
    BatchNormalization(name='readout_bn1'),
    Flatten(name='flatten'),
    Dense(fc_size, name='fc'),
    Dropout(dropout),
    Dense(37, name='predictions')
    ]

    layers.extend(readout)

    m = Sequential(layers)
    m.compile(optimizer='adam', loss='mse')
    return m

def gen_y_fake(y, sem_y):
    loc = np.zeros_like(y)
    z = np.random.normal(loc,sem_y)
    return (y + z)

def pairwise_pcc(y,y_pred):
    ppcc = [ np.corrcoef(y_pred[:,i],y[:,i]) for i in np.arange(37)]
    return np.nan_to_num(np.array(ppcc)[:,1,0])

def train_test(idxs, frac):
    # Randomize indices and partition
    randomized_idxs = np.random.permutation(idxs)
    c = round(len(idxs)*frac)
    train_idxs = randomized_idxs[:c]
    valid_idxs = randomized_idxs[c:]

    return (train_idxs, valid_idxs)

def eval_network(kfold_sets, layer_name, activity):
    y_pred_list = []
    ppcc_list = []
    fev_list = []
    ppcc_baseline_list = []
    images = [ ki.img_to_array(ki.load_img('../data/images/%g.jpg'%id, target_size=(224,224))) for id in np.arange(956) ]
    images = np.array(images)
    sess = tf.Session()
    K.set_session(sess)
    for train_idxs, valid_idxs in tqdm(kfold_sets):
        net = DeepOracle(lname=layer_name)
        train_activity = activity[train_idxs]
        valid_activity = activity[valid_idxs]

        train_images = images[train_idxs]
        valid_images = images[valid_idxs]

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)

        net.fit(train_images, train_activity,
                batch_size=32,
                epochs=30,
                verbose=0,
                # callbacks=[reduce_lr]
                )

        y_pred = net.predict(valid_images, batch_size=32)
        ppcc = pairwise_pcc(valid_activity,y_pred)
        fev_vals = fev(valid_activity, y_pred, multioutput='raw_values')
        fev_list.extend([fev_vals])
        ppcc_list.extend([ppcc])

        #y_baseline = gen_y_fake(valid_activity, sem_activity[valid_idxs])

        #ppcc_baseline = pairwise_pcc(y_baseline, valid_activity)
        #y_pred_list.extend([y_pred])
        #ppcc_baseline_list.extend([ ppcc_baseline ])

        #si = si(valid_activity)
    sess.close()

    return dict(ppcc_list=ppcc_list, fev_list=fev_list)

if __name__ == '__main__':

    # mat_file = '../data/02mean_d1.mat'
    early_file = '../data/02_stats_early.mat'
    late_file = '../data/02_stats_late.mat'
    print('loading ', early_file, '...')
    print('loading ', late_file, '...')

    e_activity_contents = sio.loadmat(early_file)
    l_activity_contents = sio.loadmat(late_file)

    e_activity = e_activity_contents['resp_mean'].swapaxes(0,1)
    l_activity = l_activity_contents['resp_mean'].swapaxes(0,1)

    # images = mat_contents['images']
    train_frac = 0.8

    # All images
    # idxs = np.arange(956)

    # Small Natural Images
    # idxs = np.arange(540)[::2]

    # Small Natural Images and gratings
    # idxs = np.arange(540)[::2]
    #idxs = np.concatenate([idxs, np.arange(540,732)])

    # Large Natural Images
    idxs = np.arange(540)[::2]

    kfold_sets = [ train_test(idxs, train_frac) for _ in np.arange(5) ]
    target_scale = (956,56,56,128)

    #base_model_layers = [ layer.name for layer in base_model.layers[1:-5] ]

    # DeepGaze II layers

    tails = [
        ['block1_conv2'],
        ['block2_conv2'],
        ['block3_conv4']
        ]
    bottom_layers = [
        ['block1_conv1'],
        ['block1_conv2'],
        ['block2_conv1'],
        ['block2_conv2']
        ]
    block3 = [
        ['block3_conv1'],
        ['block3_conv2'],
        ['block3_conv3'],
        ['block3_conv4']
    ]

    block4 = [
        ['block4_conv1'],
        ['block4_conv2'],
        ['block4_conv3'],
        ['block4_conv4']
    ]

    block5 = [
        ['block5_conv1'],
        ['block5_conv2'],
        ['block5_conv3'],
        ['block5_conv4']
        ]
    #DG2 = np.array(base_model_layers)[[16, 17, 19]]

    use_layers = []
    use_layers.extend(bottom_layers)
    use_layers.extend(block3)
    use_layers.extend(block4)
    use_layers.extend(block5)
    use_layers = use_layers[1:-1]

    early_results = []
    late_results = []
    for block in tqdm(use_layers, unit='layer'):
        up_to_layer= block[0]
        print('starting...')
        print('building model...')

        print('evaluating model on early activity for ', up_to_layer)
        early_eval_metrics = eval_network(kfold_sets, up_to_layer, e_activity)

        print('evaluating model on late activity for ', up_to_layer)
        late_eval_metrics = eval_network(kfold_sets, up_to_layer, l_activity)

        early_eval_metrics['network']=block
        late_eval_metrics['network']=block

        early_results.extend([ early_eval_metrics ])
        late_results.extend([ late_eval_metrics ])

        print('overwriting early results...')
        joblib.dump(early_results, 'tmp/early_sm_all_layers.pkl')

        print('overwriting late results...')
        joblib.dump(late_results, 'tmp/late_sm_all_layers.pkl')
