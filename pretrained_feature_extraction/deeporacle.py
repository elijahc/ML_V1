import scipy.io as sio
import numpy as np
import h5py
import gc
from tqdm import tqdm
import cv2
from keras_tqdm import TQDMCallback
from vgg19 import VGG19
from keras.preprocessing import image as ki
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
from sklearn.base import BaseEstimator
from sklearn.metrics import explained_variance_score as fev
import csv

def DeepOracle(target_shape=(14,14,512*3)):

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (num_feat, None, None)
    else:
        input_shape = (224, 224, 3)
        # input_shape = np.squeeze(activation_input).shape

    # Convolution Architecture
    # Block 1
    model = Sequential()
    model.add(Convolution2D(16, (1, 1), activation='relu', padding='same',
        name='block1_conv1', input_shape=target_shape))
    model.add(BatchNormalization(name='block1_bn1'))

    model.add(Convolution2D(32, (1, 1), activation='relu', padding='same',
        name='block1_conv2'))
    model.add(BatchNormalization(name='block1_bn2'))

    model.add(Convolution2D(2, (1, 1), activation='relu', padding='same', name='block1_conv3'))
    model.add(BatchNormalization(name='block1_bn3'))

    model.add(Convolution2D(1, (1, 1), activation='relu', padding='same', name='block1_conv4'))

    # Block 2
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', kernel_initializer='glorot_normal', name='fc1'))
    model.add(Dense(2048, activation='relu', kernel_initializer='glorot_normal', name='fc2'))
    model.add(Dense(37, activation='relu', kernel_initializer='glorot_normal', name='predictions'))

    return model

def get_activation(base_model, layer):
        return Model(input=base_model.input, output=base_model.get_layer(layer).output)

def get_activations(base_model, layers):

    activations = []
    for layer in layers:
        activations.extend([ get_activation(base_model, layer) ])

    return activations


def train_test(idxs, frac):
    # Randomize indices and partition
    randomized_idxs = np.random.permutation(idxs)
    c = round(len(idxs)*frac)
    train_idxs = randomized_idxs[:c]
    valid_idxs = randomized_idxs[c:]

    return (train_idxs, valid_idxs)

def build_random(using=None, choose=3, target_scale=None):
    # layers = np.random.choice(base_model_layers[-14:],3,replace=False)
    layers = np.random.choice(using,choose,replace=False)

    return (layers, build(layers=layers, target_scale=target_scale))

def build(layers=None, target_scale=None):

    base_model = VGG19(weights='imagenet')
    print('extracting layers:')
    print(layers)

    f = h5py.File('../data/02activations.hdf5', 'r+')
    activations = []
    for layer in layers:
        layer_activation = []
        try:
            print('extracting ',layer, ' from cache...')
            layer_activation = f['activations/'+layer][:]
        except:
            print(layer,' not in cache, rebuilding from source...')
            images = [ ki.img_to_array(ki.load_img('../data/images/%g.jpg'%id, target_size=(224,224))) for id in np.arange(956) ]
            images = np.array(images)

            activation_fetcher = get_activation(base_model, layer)
            layer_activation = activation_fetcher.predict(images,batch_size=32,verbose=1)
            num_imgs = layer_activation.shape[0]
            num_features = layer_activation.shape[3]
            sc_fac = tuple(list(np.array([num_imgs, target_scale[0], target_scale[1], num_features])/np.array(layer_activation.shape)))
            print('Rescaling by factor: ', sc_fac)
            print('resizing feature map...')
            layer_activation = zoom(layer_activation, sc_fac, order=0)
            # for img in tqdm(images):
            #     img = np.expand_dims(img, axis=0)
            #     layer_activation.extend([ feature ])

            # layer_activation = np.concatenate(layer_activation, axis=0)
            print(layer_activation.shape)
            print('caching ',layer,'...')
            f.create_dataset('activations/'+layer, data=layer_activation)
            del images
            pass

        activations.extend([ layer_activation ])

    f.close()
    del f, layer_activation
    gc.collect()

    activations = np.concatenate(activations, axis=3)

    return activations

def eval_network(kfold_sets, activations):
    y_pred_list = []
    ppcc_list = []
    ppcc_baseline_list = []
    for train_idxs, valid_idxs in kfold_sets:
        train_activity = activity[train_idxs]
        valid_activity = activity[valid_idxs]

        train_activations = activations[train_idxs]
        valid_activations = activations[valid_idxs]


        model = DeepOracle(activations.shape[1:])

        model.compile(
                optimizer='adam',
                loss='mse')

        model.fit(train_activations, train_activity, batch_size=32, nb_epoch=20)

        y_pred = model.predict(valid_activations, batch_size=32)
        ppcc = pairwise_pcc(valid_activity,y_pred)
        ppcc_list.extend([ppcc])

        #y_baseline = gen_y_fake(valid_activity, sem_activity[valid_idxs])

        #ppcc_baseline = pairwise_pcc(y_baseline, valid_activity)
        #y_pred_list.extend([y_pred])
        #ppcc_baseline_list.extend([ ppcc_baseline ])

        #si = si(valid_activity)

    return ppcc_list

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

    # All images
    # idxs = np.arange(956)

    # Small Natural Images
    # idxs = np.arange(540)[::2]

    # Small Natural Images and gratings
    idxs = np.arange(540)[::2]
    idxs = np.concatenate([idxs, np.arange(540,732)])

    kfold_sets = [ train_test(idxs, train_frac) for _ in np.arange(5) ]
    target_scale = (956,56,56,128)

    base_model_layers = [ layer.name for layer in base_model.layers[1:-5] ]

    # DeepGaze II layers

    tails = ['block1_conv2', 'block2_conv2', 'block3_conv4']
    block3 = ['block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4', 'block3_pool']
    block4 = ['block4_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv4', 'block4_pool']
    DG2 = np.array(base_model_layers)[[16, 17, 19]]

    results = []

    for block in [tails]:
        layers, activations = build_random( using=block, choose=3, target_scale=target_scale)

        ppcc_list = eval_network(kfold_sets, activations)
        results.extend([ {'network': layers, 'ppcc_list': ppcc_list}])

    import pdb; pdb.set_trace()

    with open('ppcc_block3.csv', 'w') as csvfile:
        fieldnames = ['ppcc','ppcc_baseline','neuron']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ppcc, baseline in zip(ppcc_list, ppcc_baseline_list):
            for n in np.arange(37):
                writer.writerow({
                    'ppcc':ppcc[n],
                    'ppcc_baseline':ppcc_baseline[n],
                    'neuron': n
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
