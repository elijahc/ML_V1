import pandas as pd
import numpy as np
from neon.data.datasets import Dataset
from v1_lstm import V1HDFSTimeSeries, V1IteratorSequence

class KohnV1Dataset(Dataset):

    def __init__(self,
                 filename='kohn.h5.tar.gz',
                 url='https://www.dropbox.com/s/l6ufcsvcnuqssx0/',
                 path='~/nervana/datasets/kohn_v1/',
                 subset='02',
                 subset_pct=80.0):

        self.filename = filename
        self.url = url
        #self.size = size
        self.path = path
        self.subset_pct = subset_pct
        self.subset = subset
        # Assumed data already exists where its supposed to
        self.store = pd.HDFStore(self.path+'kohn.h5')

    def get_timeseries(self, subset, binning):
        df = self.load_subset(subset)
        self.ts = V1HDFSTimeSeries(df, self.time_steps, binning=binning, divide=(1-self.subset_pct))
        return self.ts

    def get_iterator(self, setname):
        ts_set = self.ts.get_set(setname)
        return V1IteratorSequence(ts_set, self.time_steps, return_sequences=False)

    def gen_iterators(self, time_steps, subset='02', binning='20ms'):
        self.time_steps = time_steps

        self.ts = self.get_timeseries(subset, binning)

        self.test_iter = self.get_iterator('test')
        self.train_iter = self.get_iterator('train')
        self.train_valid = self.get_iterator('valid')

        #self.data_set = {
        #    'test': self.test_iter,
        #    'train': self.train_iter,
        #    'valid': self.train_iter}

        return self.data_set

    def load_subset(self, subset):
        prefix = 'raw_' + str(subset)+'/'
        target = prefix+'n_slugs'

        # Fetch list of neurons time traces e.g. store['raw_02/n_slugs']
        n_slugs = self.store[target]
        neurons = [self.store[s] for s in n_slugs]
        stim = self.store[prefix+'stim']

        # Merge into a single data frame
        df = pd.DataFrame()
        cols = [df]
        cols.extend([stim])
        cols.extend(neurons)
        self.n_df = pd.concat(cols, axis=1)
        names = self.n_df.columns.tolist()
        names[0] = 'stim'
        self.n_df.columns = names

        return self.n_df
