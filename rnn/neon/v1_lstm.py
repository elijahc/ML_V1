from builtins import object
import numpy as np
import math
import pandas as pd
from neon.data.dataiterator import ArrayIterator
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

class V1HDFSTimeSeries(object):

    def __init__(self, df, time_steps, binning='10ms', divide=0.2, scale=False):

        self.n_df = df
        self.time_steps = time_steps
        self.train_set = None
        self.test_set = None

        # Remap index to date-times
        di = pd.date_range('1/1/2000', periods=len(self.n_df), freq='L')
        self.n_df.index = di

    def get_set(self,setname):
        if self.train_set or self.test_set == None:
            self.segregate()

        if setname == 'test':
            return self.test_set
        elif setname == 'train':
            return self.train_set
        else:
            return None

    def test_set(self):
        if self.test_set == None:
            self.segregate()

        return self.test_set

    def segregate(self, binning='20ms', divide=0.2):

        # Do Scaling later I guess?
        # scaler = MinMaxScaler(feature_range=(0,1))
        # if scale:
        #     self.data = scaler.fit_transform(self.data)

        # Resample according to binning summing across bins
        n_df_binned = self.n_df.resample(binning).sum()
        self.n_samples = len(n_df_binned.index)

        # Just straight split them for now, randomize later
        c = int(self.n_samples * (divide))
        test_set = {'stim': n_df_binned['stim'][:c],
                    'spikes': np.nan_to_num( n_df_binned.drop('stim', axis=1)[:c].as_matrix() )}
        train_set = {'stim': n_df_binned['stim'][c:],
                     'spikes': np.nan_to_num( n_df_binned.drop('stim',axis=1)[c:].as_matrix() )}

        self.train_set = train_set
        self.test_set = test_set


class V1TimeSeries(object):

    def __init__(self, spikes, stim, binning=10, divide=0.2, scale=False):
        self.spikes = TimeSeries(x=spikes, binning=binning, divide=divide, scale=False)
        self.stim = TimeSeries(x=stim.reshape(-1,1), binning=binning, divide=divide, scale=False)

        self.train = {'spikes':self.spikes.train, 'stim':self.stim.train}
        self.test = {'spikes':self.spikes.test, 'stim':self.stim.test}

class V1IteratorSequence(NervanaObject):

    """
    This class takes a sequence and returns an iterator providing data in batches suitable for RNN
    prediction.  Meant for use when the entire dataset is small enough to fit in memory.
    """

    def __init__(self, X_dict, time_steps, forward=1, return_sequences=True, randomize=False):
        """
        Implements loading of given data into backend tensor objects. If the backend is specific
        to an accelerator device, the data is copied over to that device.

        Args:
            X (ndarray): Input sequence with feature size within the dataset.
                         Shape should be specified as (num examples, feature size]
            time_steps (int): The number of examples to be put into one sequence.
            forward (int, optional): how many forward steps the sequence should predict. default
                                     is 1, which is the next example
            return_sequences (boolean, optional): whether the target is a sequence or single step.
                                                  Also determines whether data will be formatted
                                                  as strides or rolling windows.
                                                  If true, target value be a sequence, input data
                                                  will be reshaped as strides.  If false, target
                                                  value will be a single step, input data will be
                                                  a rolling_window
        """

        def rolling_window(spikes,stim, lag):
            """
            Convert a into time-lagged vectors

            a    : (n, p)
            lag  : time steps used for prediction

            returns  (n-lag+1, lag, p)  array

            (Building time-lagged vectors is not necessary for neon.)
            """
            assert spikes.shape[0] > lag
            assert stim.shape[0] > lag

            spikes_shape = [spikes.shape[0] - lag + 1, lag, spikes.shape[-1]]
            stim_shape = [stim.shape[0] - lag + 1, lag, stim.shape[-1]]
            spikes_strides = [spikes.strides[0], spikes.strides[0], spikes.strides[-1]]
            stim_strides = [stim.strides[0], stim.strides[0], stim.strides[-1]]

            spikes_out = np.lib.stride_tricks.as_strided(spikes, shape=spikes_shape, strides=spikes_strides)
            stim_out = np.lib.stride_tricks.as_strided(stim, shape=stim_shape, strides=stim_strides)
            return {'spikes': spikes_out, 'stim':stim_out}

        self.seq_length = time_steps
        self.forward = forward
        self.batch_index = 0
        self.nfeatures = self.nclass = X_dict['spikes'].shape[1]
        self.nsamples = X_dict['spikes'].shape[0]
        self.shape = (self.nfeatures, time_steps)
        self.return_sequences = return_sequences
        self.mean = X_dict['spikes'].mean(axis=0)

        target_steps = time_steps if return_sequences else 1
        # pre-allocate the device buffer to provide data for each minibatch
        # buffer size is nfeatures x (times * batch_size), which is handled by
        # backend.iobuf()
        self.X_dev = self.be.iobuf((self.nfeatures, time_steps))
        self.y_dev = self.be.iobuf((self.nfeatures, target_steps))

        if return_sequences is True:
            # truncate to make the data fit into multiples of batches
            extra_examples = self.nsamples % (self.be.bsz * time_steps)
            if extra_examples:
                X_dict['spikes'] = X_dict['spikes'][:-extra_examples]
                X_dict['stim'] = X_dict['stim'][:-extra_examples]

            # calculate how many batches
            self.nsamples -= extra_examples
            self.nbatches = self.nsamples // (self.be.bsz * time_steps)
            self.ndata = self.nbatches * self.be.bsz * time_steps  # no leftovers

            # y is the lagged version of X
            y = np.concatenate((X[forward:], X[:forward]))
            self.y_series = y
            # reshape this way so sequence is continuous along the batches
            self.X = X.reshape(self.be.bsz, self.nbatches,
                               time_steps, self.nfeatures)
            self.y = y.reshape(self.be.bsz, self.nbatches,
                               time_steps, self.nfeatures)
        else:
            X_dict['lag'] = time_steps
            self.X = rolling_window(**X_dict)
            self.X['spikes'] = self.X['spikes'][:-1]
            self.X['stim'] = self.X['stim'][:-1]
            self.y = X_dict['spikes'][time_steps:]

            self.nsamples = self.X['spikes'].shape[0]
            extra_examples = self.nsamples % (self.be.bsz)
            if extra_examples:
                self.X['spikes'] = self.X['spikes'][:-extra_examples]
                self.X['stim'] = self.X['stim'][:-extra_examples]
                self.y = self.y[:-extra_examples]

            # calculate how many batches
            self.nsamples -= extra_examples
            self.nbatches = self.nsamples // self.be.bsz
            self.ndata = self.nbatches * self.be.bsz
            self.y_series = self.y
            self.spike_series = self.X['spikes']
            self.stim_series = self.X['stim']

            Xshape = (self.nbatches, self.be.bsz, time_steps, self.nfeatures)
            Yshape = (self.nbatches, self.be.bsz, 1, self.nfeatures)
            #self.X = self.X.reshape(Xshape).transpose(1, 0, 2, 3)
            #self.y = self.y.reshape(Yshape).transpose(1, 0, 2, 3)

            return ArrayIterator(X=[self.spike_series, self.stim_series],y=self.y_series,make_onehot=False)

    # def reset(self):
    #     """
    #     For resetting the starting index of this dataset back to zero.
    #     """
    #     self.batch_index = 0

    # def __iter__(self):
    #     """
    #     Generator that can be used to iterate over this dataset.

    #     Yields:
    #         tuple : the next minibatch of data.
    #     """
    #     self.batch_index = 0
    #     while self.batch_index < self.nbatches:
    #         # get the data for this batch and reshape to fit the device buffer
    #         # shape
    #         X_batch = self.X[:, self.batch_index].T.reshape(
    #             self.X_dev.shape).copy()
    #         y_batch = self.y[:, self.batch_index].T.reshape(
    #             self.y_dev.shape).copy()

    #         # make the data for this batch as backend tensor
    #         self.X_dev.set(X_batch)
    #         self.y_dev.set(y_batch)

    #         self.batch_index += 1

    #         yield self.X_dev, self.y_dev
