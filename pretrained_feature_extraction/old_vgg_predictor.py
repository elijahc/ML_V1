import scipy.io as sio
import numpy as np
import h5py
from tqdm import tqdm
import cv2
from keras_tqdm import TQDMCallback
from vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Lambda
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from imagenet_utils import preprocess_input
from scipy.ndimage.interpolation import zoom
from skimage.color import gray2rgb
from skimage.io import imread
from skimage.transform import resize
import csv

def DeepOracle(layers, input_tensor=None):

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (num_feat, None, None)
    else:
        input_shape = (224, 224, 3)
        # input_shape = np.squeeze(activation_input).shape

    # Convolution Architecture
    # Block 1
    model = Sequential()
    model.add(Convolution2D(16, 1, 1, activation='relu', border_mode='same', name='block1_conv1', input_shape=(14,14,3*512)))
    model.add(BatchNormalization(name='block1_bn1'))
    model.add(Convolution2D(32, 1, 1, activation='relu', border_mode='same', name='block1_conv2'))
    model.add(BatchNormalization(name='block1_bn2'))
    model.add(Convolution2D(2, 1, 1, activation='relu', border_mode='same', name='block1_conv3'))
    model.add(BatchNormalization(name='block1_bn3'))
    model.add(Convolution2D(1, 1, 1, activation='relu', border_mode='same', name='block1_conv4'))

    # Block 2
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', init='glorot_normal', name='fc1'))
    model.add(Dense(2048, activation='relu', init='glorot_normal', name='fc2'))
    model.add(Dense(37, activation='relu', init='glorot_normal', name='predictions'))

    return model

def get_activations(base_model, layers):

    activations = []

    for layer in layers:

        model = Model(input=base_model.input, output=base_model.get_layer(layer).output)
        activations.extend([ model ])

    return activations

def pairwise_pcc(y,y_pred):
    ppcc = [ np.corrcoef(y_pred[:,i],y[:,i]) for i in np.arange(37)]
    return np.nan_to_num(np.array(ppcc)[:,1,0])


if __name__ == '__main__':

    mat_file = '../data/02mean_d1.mat'
    print('loading mat data...', mat_file)
    mat_contents = sio.loadmat(mat_file)
    activity = mat_contents['activity']

    # images = mat_contents['images']
    train_frac = 0.8

    base_model = VGG19(weights='imagenet')
    base_model_layers = [ layer.name for layer in base_model.layers[1:-5] ]
    layers = np.array(base_model_layers)[[16, 17, 19]]
    print('extracting layers:')
    print(layers)

    idxs = np.random.permutation(np.arange(956))
    c = round(956*train_frac)
    train_idxs = idxs[:c]
    valid_idxs = idxs[c:]

    train_activity = activity[train_idxs]
    valid_activity = activity[valid_idxs]

    try:
        f = h5py.File('../data/02activations.hdf5', 'r')
        activations = f['activations'][:]
        f.close()
    except:
        images = [ cv2.resize(cv2.imread('../data/images/%g.jpg'%id),(224,224)).astype(np.float32) for id in tqdm(np.arange(956),desc='loading images') ]
        images = np.array(images)

        train_images = images[train_idxs]
        valid_images = images[valid_idxs]

        activation_fetchers = get_activations(base_model, layers)
        for img in tqdm(images):
            img = np.expand_dims(img, axis=0)
            features = [ feature.predict(img) for feature in activation_fetchers ]
            features = np.concatenate(features, axis=3)
            activations.extend([ features ])

        activations = np.concatenate(activations, axis=0)
        f = h5py.File('../data/02activations.hdf5', 'w')
        f.create_dataset('activations', data=activations)
        f.close()
        pass
    train_activations = activations[train_idxs]
    valid_activations = activations[valid_idxs]

    model = DeepOracle(layers)

    model.compile(
            optimizer='rmsprop',
            loss='mse',
            metrics=[])

    model.fit(train_activations, train_activity, batch_size=32, nb_epoch=10)
    y_pred = model.predict(valid_activations, batch_size=32)
    ppcc = pairwise_pcc(valid_activity,y_pred)
    with open('ppcc.csv', 'w') as csvfile:
        fieldnames = ['ppcc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in np.arange(len(ppcc)):
            writer.writerow({'ppcc':ppcc[i]})

    with open('evaluation.csv', 'w') as csvfile:
        fieldnames = ['id','y', 'y_pred', 'neuron']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i,idx in enumerate(valid_idxs):
            for n in np.arange(37):
                writer.writerow({'id':idx, 'y': valid_activity[i,n], 'y_pred':y_pred[i,n], 'neuron':n})
    print('pcc: %.3f' % ppcc)
