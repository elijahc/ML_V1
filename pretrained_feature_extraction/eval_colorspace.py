#from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from imagenet_utils import decode_predictions, preprocess_input
from vgg19 import VGG19
import pprint

def eval(paths=['cat.jpg']):

    model = VGG19(include_top=True, weights='imagenet')

    imgs = [ image.load_img(img_path, target_size=(224,224)) for img_path in paths ]

    preds = []

    for img in imgs:
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        print('Input image shape:', x.shape)

        preds.extend(decode_predictions(model.predict(x)))

    return preds

def ar_to_dict(predictions):
    out_dict = {}
    for elem in predictions:
        out_dict[ elem[1] ]=elem[2]

    return out_dict

def rdm(vec1,vec2):
    num = np.cov(vec1, vec2)
    denom = np.sqrt(np.var(vec1) * np.var(vec2))

    return 1 - num/denom

pp = pprint.PrettyPrinter(indent=2)
img_path = 'cat.jpg'
grey_img_path = 'grey_cat.jpg'
results = eval([img_path, grey_img_path])
results = [ ar_to_dict(val) for val in results ]
cons = [ sorted(list(res.values()), reverse=True) for res in results ]
pp.pprint(results)

import pdb; pdb.set_trace()
