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
from keras import backend as K
from imagenet_utils import preprocess_input
from scipy.ndimage.interpolation import zoom
from selectivity import si
from sklearn.base import BaseEstimator
from sklearn.metrics import explained_variance_score as fev
import csv

def DeepOracle(target_shape=(14,14,512*3)):

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (num_feat, None, None)
    else:
        input_shape = target_shape
        # input_shape = np.squeeze(activation_input).shape

    # Convolution Architecture
    # Block 1
    model = Sequential()
    # model.add(Convolution2D(2, (1, 1), activation='relu', padding='same',
    #     name='block1_conv1', input_shape=target_shape))
    # model.add(BatchNormalization(name='block1_bn1'))

    model.add(Convolution2D(1, (1, 1), activation='relu', padding='same',
        name='block1_conv1', input_shape=target_shape))
    model.add(BatchNormalization(name='block1_bn1'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(300, name='fc'))
    model.add(Dropout(0.5))
    model.add(Dense(37, name='predictions'))

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
            layer_activation = activation_fetcher.predict(images,batch_size=32,verbose=1).astype(np.float16)

            # if rescaling uncomment this

            # num_imgs = layer_activation.shape[0]
            # num_features = layer_activation.shape[3]
            # sc_fac = tuple(list(np.array([num_imgs, target_scale[0], target_scale[1], num_features])/np.array(layer_activation.shape)))
            # print('Rescaling by factor: ', sc_fac)
            # print('resizing feature map...')
            # layer_activation = zoom(layer_activation, sc_fac, order=0)
            # for img in tqdm(images):
            #     img = np.expand_dims(img, axis=0)
            #     layer_activation.extend([ feature ])

            # layer_activation = np.concatenate(layer_activation, axis=0)
            print(layer_activation.shape)
            print('caching ',layer,'...')
            f.create_dataset('activations/'+layer, data=layer_activation)
            del images
            pass

        activations.extend([ layer_activation.astype(np.float32) ])

    f.close()
    # del f, layer_activation
    # gc.collect()

    activations = np.concatenate(activations, axis=3)

    return activations

def eval_network(kfold_sets, activations, activity):
    y_pred_list = []
    ppcc_list = []
    fev_list = []
    ppcc_baseline_list = []
    for train_idxs, valid_idxs in tqdm(kfold_sets, desc='kfold', leave=False):
        train_activity = activity[train_idxs]
        valid_activity = activity[valid_idxs]

        train_activations = activations[train_idxs]
        valid_activations = activations[valid_idxs]


        mod = DeepOracle(activations.shape[1:])

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
        mod.compile(
                optimizer='adam',
                loss='mse',
                metrics=[])

        mod.fit(train_activations, train_activity,
                batch_size=32,
                epochs=20,
                verbose=0
                )

        y_pred = mod.predict(valid_activations, batch_size=32)
        ppcc = pairwise_pcc(valid_activity,y_pred)
        fev_vals = fev(valid_activity, y_pred, multioutput='raw_values')
        fev_list.extend([fev_vals])
        ppcc_list.extend([ppcc])

        #y_baseline = gen_y_fake(valid_activity, sem_activity[valid_idxs])

        #ppcc_baseline = pairwise_pcc(y_baseline, valid_activity)
        #y_pred_list.extend([y_pred])
        #ppcc_baseline_list.extend([ ppcc_baseline ])

        #si = si(valid_activity)

    return dict(ppcc_list=ppcc_list, fev_list=fev_list)

if __name__ == '__main__':
    try:
        tf.InteractiveSession()
    except:
        pass

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
    idxs = np.arange(540)[::2]
    idxs = np.concatenate([idxs, np.arange(540,732)])

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
        activations = build(block, target_scale=(112,112))

        print('evaluating model on early activity for ', block)
        early_eval_metrics = eval_network(kfold_sets, activations, e_activity)

        print('evaluating model on late activity for ', block)
        late_eval_metrics = eval_network(kfold_sets, activations, l_activity)

        early_eval_metrics['network']=block
        late_eval_metrics['network']=block

        early_results.extend([ early_eval_metrics ])
        late_results.extend([ late_eval_metrics ])

        print('overwriting early results...')
        joblib.dump(early_results, 'tmp/early_all_layers.pkl')

        print('overwriting late results...')
        joblib.dump(late_results, 'tmp/late_all_layers.pkl')
