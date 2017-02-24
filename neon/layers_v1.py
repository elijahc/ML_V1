import numpy as np
import math
from neon.transforms.cost import Metric, Cost
from neon.transforms.transform import Transform
from neon.layers.layer import Layer

class RoundInt(Transform):

    def __init__(self, name=None):
        super(RoundInt,self).__init__(name)

    def __call__(self,x):
        rounded_int_val = self.be.rint(x)
        import pdb; pdb.set_trace()
        return rounded_int_val

class Round(Layer):

    def __init__(self, name=None):
        super(Round,self).__init__(name)

    def configure(self,in_obj):
        super(Round,self).configure(in_obj)
        self.out_shape = self.in_shape
        return self

    def fprop(self,inputs,inference=False):
        self.outputs = self.be.array(inputs.get().round())

        return self.outputs

    def bprop(self,error):
        error[:] = error
        return error
