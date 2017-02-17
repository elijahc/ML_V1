from builtins import object
import numpy as np
import math
from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.layers import GeneralizedCost, LSTM, Affine, RecurrentLast
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Logistic, Tanh, Identity, MeanSquared
from neon.callbacks.callbacks import Callbacks
from neon import NervanaObject, logger as neon_logger
from neon.util.argparser import NeonArgparser, extract_valid_args

class TimeSeries(object):

    def __init__(self, x, binning=10, divide=0.2):
        self.x = x

        self.nfeatures = np.size(self.x,axis=1)
        self.data = x.reshape(binning, -1, self.nfeatures).sum(axis=0)

        L = len(self.data)
        c = int(L * (1 - divide))
        self.train = self.data[:c]
        self.test = self.data[c:]


class DataIteratorSequence(NervanaObject):

    """
    This class takes a sequence and returns an iterator providing data in batches suitable for RNN
    prediction.  Meant for use when the entire dataset is small enough to fit in memory.
    """

    def __init__(self, X, time_steps, forward=1, return_sequences=True):
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
        self.seq_length = time_steps
        self.forward = forward
        self.batch_index = 0
        self.nfeatures = self.nclass = X.shape[1]
        self.nsamples = X.shape[0]
        self.shape = (self.nfeatures, time_steps)
        self.return_sequences = return_sequences

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
                X = X[:-extra_examples]

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
            self.X = rolling_window(X, time_steps)
            self.X = self.X[:-1]
            self.y = X[time_steps:]

            self.nsamples = self.X.shape[0]
            extra_examples = self.nsamples % (self.be.bsz)
            if extra_examples:
                self.X = self.X[:-extra_examples]
                self.y = self.y[:-extra_examples]

            # calculate how many batches
            self.nsamples -= extra_examples
            self.nbatches = self.nsamples // self.be.bsz
            self.ndata = self.nbatches * self.be.bsz
            self.y_series = self.y

            Xshape = (self.nbatches, self.be.bsz, time_steps, self.nfeatures)
            Yshape = (self.nbatches, self.be.bsz, 1, self.nfeatures)
            self.X = self.X.reshape(Xshape).transpose(1, 0, 2, 3)
            self.y = self.y.reshape(Yshape).transpose(1, 0, 2, 3)

    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        """
        self.batch_index = 0

    def __iter__(self):
        """
        Generator that can be used to iterate over this dataset.

        Yields:
            tuple : the next minibatch of data.
        """
        self.batch_index = 0
        while self.batch_index < self.nbatches:
            # get the data for this batch and reshape to fit the device buffer
            # shape
            X_batch = self.X[:, self.batch_index].T.reshape(
                self.X_dev.shape).copy()
            y_batch = self.y[:, self.batch_index].T.reshape(
                self.y_dev.shape).copy()

            # make the data for this batch as backend tensor
            self.X_dev.set(X_batch)
            self.y_dev.set(y_batch)

            self.batch_index += 1

            yield self.X_dev, self.y_dev
