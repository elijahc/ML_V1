from deeporacle import build
import numpy as np

def img_ary(data):
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

    return rescaled

layers = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']

outputs = build(layers, (112,112))

ref = img_ary(outputs[0,:,:,0])
import pdb; pdb.set_trace()
