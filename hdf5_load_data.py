import pandas as pd
import numpy as np
import scipy.io as sio
import ipdb, sys, traceback
import pickle

store_path = 'tmp/kohn.h5'
store = pd.HDFStore(store_path)

FILES = ['02', '10', '08', '05']

for f in FILES:
    fp = 'data/timeseries/'+f+'_timeseries.mat'

    print('loading file...'+fp)
    matfile = sio.loadmat(fp)
    timeseries, stim, trial_num, image_id = (matfile['timeseries'], matfile['stim'], matfile['trial_num'],matfile['image_id'])
    nfeatures = timeseries.shape[1]
    raw_samples = timeseries.shape[0]
    print('Neurons: %g  total time (ms): %g' % (nfeatures, raw_samples))

    print('extracting neuron firing vectors...')
    #ns_vec = [np.nonzero(c)[1] for c in np.vsplit(timeseries, timeseries.shape[0])]
    time_vec = np.arange(raw_samples)
    non_zero_spike_idxs = []
    dense_spike_trains = []
    for n in np.arange(nfeatures):
        idxs = np.nonzero(timeseries[:,n])
        non_zero_spike_idxs.extend([idxs])
        dense_spike_trains.extend([ (time_vec[idxs],np.squeeze(timeseries[idxs,n])) ])

    ts_vec = [pd.Series(s,index=t) for i,(t,s) in enumerate(dense_spike_trains)]
    print('adding '+f+' data to '+store_path)

    slugs = []
    for i,ts in enumerate(ts_vec):
        slug = 'raw_'+f+'/n'+str(i)
        slugs.extend([ slug ])
        store.put(slug, ts)

    store.put('raw_'+f+'/n_slugs', pd.Series(slugs))
    store.put('raw_'+f+'/stim', pd.Series(np.squeeze(stim)))
    store.put('raw_'+f+'/trial_num', pd.Series(np.squeeze(trial_num)))
    store.put('raw_'+f+'/image_id', pd.Series(np.squeeze(image_id)))

    #df.index.name = 'time(ms)'

    #df.to_pickle('tmp/'+f+'_timeseries.pck')
#df.to_pickle('tmp/08_ledger.pck')
store.close()
