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
from selectivity import si
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
    model.add(Convolution2D(16, 1, 1, activation='relu', border_mode='same', name='block1_conv1', input_shape=(28,28,3*512)))
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

def get_activation(base_model, layer):
        return Model(input=base_model.input, output=base_model.get_layer(layer).output)

def get_activations(base_model, layers):

    activations = []
    for layer in layers:
        activations.extend([ get_activation(base_model, layer) ])

    return activations

def gen_y_fake(y, sem_y):
    loc = np.zeros_like(y)
    z = np.random.normal(loc,sem_y)
    return (y + z)

def pairwise_pcc(y,y_pred):
    ppcc = [ np.corrcoef(y_pred[:,i],y[:,i]) for i in np.arange(37)]
    return np.nan_to_num(np.array(ppcc)[:,1,0])


if __name__ == '__main__':

    mat_file = '../data/02mean_d1.mat'
    activity_file = '../data/02_stats.mat'
    print('loading mat data...', mat_file)
    mat_contents = sio.loadmat(mat_file)
    activity_contents = sio.loadmat(activity_file)
    activity = activity_contents['resp_mean'].swapaxes(0,1)
    sem_activity = activity_contents['resp_sem'].swapaxes(0,1)

    # images = mat_contents['images']
    train_frac = 0.8

    base_model = VGG19(weights='imagenet')
    base_model_layers = [ layer.name for layer in base_model.layers[1:-5] ]
    block4 = ['block4_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv4', 'block4_pool']

    # DeepGaze II layers
    # layers = np.array(base_model_layers)[[16, 17, 19]]

    # layers = np.random.choice(base_model_layers[-9:],3,replace=False)
    layers = np.random.choice(block4,3,replace=False)

    print('extracting layers:')
    print(layers)

    # All images
    # idxs = np.arange(956)

    # Small Natural Images
    # idxs = np.arange(540)[::2]

    # Small Natural Images and gratings
    idxs = np.arange(540)[::2]
    idxs = np.concatenate([idxs, np.arange(540,732)])


    # Randomize indices and partition
    randomized_idxs = np.random.permutation(idxs)
    c = round(len(idxs)*train_frac)
    train_idxs = randomized_idxs[:c]
    valid_idxs = randomized_idxs[c:]

    train_activity = activity[train_idxs]
    valid_activity = activity[valid_idxs]

    f = h5py.File('../data/02activations.hdf5', 'r+')
    activations = []
    target_scale = 28
    for layer in layers:
        layer_activation = []
        try:
            print('extracting ',layer, ' from cache...')
            layer_activation = f['activations/'+layer][:]
        except:
            print(layer,' not in cache, rebuilding from source...')
            images = [ cv2.resize(cv2.imread('../data/images/%g.jpg'%id),(224,224)).astype(np.float32) for id in tqdm(np.arange(956),desc='loading images') ]
            images = np.array(images)

            activation_fetcher = get_activation(base_model, layer)
            layer_activation = activation_fetcher.predict(images,batch_size=128,verbose=1)
            # for img in tqdm(images):
            #     img = np.expand_dims(img, axis=0)
            #     layer_activation.extend([ feature ])

            # layer_activation = np.concatenate(layer_activation, axis=0)
            print('caching ',layer,'...')
            f.create_dataset('activations/'+layer, data=layer_activation)
            pass
        sc_fac = target_scale//layer_activation.shape[1]
        if sc_fac > 1:
            layer_activation = zoom(layer_activation, (1,sc_fac,sc_fac,1), order=0)
        activations.extend([ layer_activation ])

    f.close()

    activations = np.concatenate(activations, axis=3)

    train_activations = activations[train_idxs]
    valid_activations = activations[valid_idxs]

    model = DeepOracle(layers)

    model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[])

    model.fit(train_activations, train_activity, batch_size=32, nb_epoch=20)
    y_pred = model.predict(valid_activations, batch_size=32)
    y_baseline = gen_y_fake(valid_activity, sem_activity[valid_idxs])

    ppcc_baseline = pairwise_pcc(y_baseline, valid_activity)
    ppcc = pairwise_pcc(valid_activity,y_pred)

    si = si(valid_activity)
    with open('ppcc.csv', 'w') as csvfile:
        fieldnames = ['ppcc','ppcc_baseline','si']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in np.arange(len(ppcc)):
            writer.writerow({
                'ppcc':ppcc[i],
                'ppcc_baseline':ppcc_baseline[i],
                'si': si[i]
                })

    with open('evaluation.csv', 'w') as csvfile:
        fieldnames = ['id','y', 'y_pred', 'neuron']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i,idx in enumerate(valid_idxs):
            for n in np.arange(37):
                writer.writerow({'id':idx, 'y': valid_activity[i,n], 'y_pred':y_pred[i,n], 'neuron':n})
    print('avg_pcc_best: %.3f' % ppcc_baseline.mean())
    print('avg_pcc: %.3f' % ppcc.mean())
    print('norm_avg_pcc: %.3f' % (ppcc/ppcc_baseline).mean())
