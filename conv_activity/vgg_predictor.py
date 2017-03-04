import numpy as np
from vgg19 import VGG19
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from scipy.ndimage.interpolation import zoom

def DeepOracle(input_tensor=None):

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (512, None, None)
    else:
        input_shape = (None, None, 512)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    x = Convolution2D(16, 1, 1, activation='relu', border_mode='same', name='block1_conv1') (img_input)
    x = Convolution2D(32, 1, 1, activation='relu', border_mode='same', name='block1_conv2') (x)
    x = Convolution2D(2, 1, 1, activation='relu', border_mode='same', name='block1_conv3') (x)
    x = Convolution2D(1, 1, 1, activation='relu', border_mode='same', name='block1_conv4') (x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(2048, activation='relu', name='fc2')(x)
    x = Dense(37, activation='relu', name='predictions')(x)


def get_activations(layers):

    activations = []
    base_model = VGG19(weights='imagenet')

    img_path = 'cat.jpg'
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)

    for layer in layers:

        model = Model(input=base_model.input, output=base_model.get_layer(layer).output)
        features = model.predict(x)

        features = zoom(features, [1, 8.0, 8.0, 1])

        activations.extend([ features ])

    import pdb; pdb.set_trace()
    for layer, features in zip(layers, activations):
        print(layer, ': ', features.shape)

    return activations


if __name__ == '__main__':
    blocks = [
            'block1_conv1',
            'block1_conv2',
            'block1_pool',
            'block2_pool',
            'block3_pool',
            'block4_pool',
            'block5_pool'
            ]
    layers = [
            # 'block2_conv1',
            'block5_conv1',
            'block5_conv2',
            'block5_conv4']
    input = get_activations(layers)
#print('features size: ',features.shape)

