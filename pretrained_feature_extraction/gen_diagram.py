from matplotlib import rc
#from old_vgg_predictor import DeepOracle
from vgg19 import VGG19
rc("font",family="serif",size=12)
rc("text", usetex=True)

import daft
class KerasLayers(object):
    def __init__(self):
        # do something here
        x = 0

    def input_layer(self, pgm):
        pgm.add_node(daft.Node("In",r"in",0.5,2,fixed=True))
        return 0

    def convolution2D(self, pgm):

        return 0

    def maxpooling2D(self, pgm):
        return 0

class NetworkDiagram(object):
    def __init__(self, model):
        self.model = model
        self.layers = self.model.layers

        self.be = KerasLayers()

    def draw_layer(self, pgm, layer):

        type = self.layer_type(layer)
        import pdb; pdb.set_trace()
        if type == 'InputLayer':
            self.be.input_layer(pgm)

    def draw_network(self):

        pgm = daft.PGM([2.3,2.05], origin=[0.3,0.3])

        for layer in self.layers:

            self.draw_layer(pgm,layer)

    def layer_type(self,layer):
        return layer.__class__.__name__


model = VGG19('imagenet')
diagram = NetworkDiagram(model)
diagram.draw_network()
