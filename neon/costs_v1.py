from builtins import object
import numpy as np
import math
from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.layers import GeneralizedCost, LSTM, Affine, RecurrentLast
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Logistic, Tanh, Identity, MeanSquared
from neon.transforms.cost import Metric, Cost
from neon.callbacks.callbacks import Callbacks
from neon import NervanaObject, logger as neon_logger
from neon.util.argparser import NeonArgparser, extract_valid_args
from sklearn.preprocessing import MinMaxScaler

class FractionExplainedVariance(Metric):

    def __init__(self):
        self.metric_names = ['FEV']
        self.batch_size = self.be.bsz
        #self.nfeatures = y.shape[0]
        #self.time_steps = self.nfeatures[1] / self.batch_size
        self.fev = self.be.iobuf(1)

    def __call__(self, y,t,calcrange=slice(0,None)):
        #self.t_mean = self.be.mean(t,axis=1)
        #self.var[:] = self.be.sum(self.be.square(self.t_mean - t), axis=0) / 2.
        #self.fev[:] = 1 - ((self.be.sum(self.be.square(y - t), axis=0) / 2.) / self.fev)

        return np.array(self.fev.get()[:,calcrange].mean())

class WeightedSumSquared(Cost):

    def __init__(self, weights):
        self.a = self.be.array(weights)
        self.funcgrad = lambda y,t: self.a*(y-t)

    def __call__(self, y,t):

        self.se = self.be.square(y-t)
        self.wse = self.be.sum(self.se*self.a,axis=0)/2.
        return self.wse
