import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description='RNN for modeling neuron populations')
    parser.add_argument('infile', metavar='infile', type=str,
                        help='Input data file path')
    parser.add_argument('outfile', metavar='outfile', type=str,
                        help='Path to output data')
    parser.add_argument('--data_structure', type=str, default='timeseries', choices=['timeseries','ledger'],
                        help='Structure to parse the data into default: timeseries')
    parser.add_argument('--format', type=str, default='mat', choices=['mat','hdf5','csv','pickle'],
                        help='File Format to save data default: mat')

    FLAGS = parser.parse_args()
    # Load files
    print('loading...stim_sequence')
    stim_sequence = sio.loadmat('data/stimulus_sequence.mat')['stimulus_sequence']
    FILE = FLAGS.infile

    print('loading...', FILE)
    session_1 = sio.loadmat(FILE)

    #%%
    # Filter out poor quality neurons
    mask = np.squeeze(session_1['INDCENT']).astype(bool)

    resp_train = session_1['resp_train'][mask]
    stim_len = np.size(resp_train,axis=-1)
    resp_train_blk = session_1['resp_train_blk'][mask]
    blank_len = np.size(resp_train_blk,axis=-1)

    # Shift by 50ms to account for response latency
    latency = 50
    resp = np.concatenate((resp_train,resp_train_blk), axis=3)
    #resp = np.roll(resp,-latency,3)[:,:,:,:-latency]


    stim, spike_train,ids,trial = mutate(resp,stim_len,blank_len,stim_sequence)

    outfile = FLAGS.outfile
    print('writing ', outfile, '...')
    sio.savemat(outfile, {'timeseries':spike_train, 'stim':stim,'trial_num':trial, 'image_id':ids})

def mutate(resp,stim_len,blank_len,stim_sequence):
    image_bin = []
    spikes = []
    image_ids = []
    trial_ids = []
    trials = np.size(resp,2)
    num_neurons = np.size(resp,0)
    num_images = np.size(resp, 1)
    i = 0
    for r in tqdm(np.arange(trials)):
        for image_id in stim_sequence[:,r]:
            index = {'i': i,
                     'trial': r,
                     'image': image_id-1
                     }
            x_on = np.zeros(stim_len, dtype=np.uint8) + 1
            x_off= np.zeros(blank_len, dtype=np.uint8) + 0
            x = np.concatenate((x_on, x_off))
            trial_vec = np.zeros_like(x,dtype=np.uint8) + r
            image_vec = np.zeros_like(x,dtype=np.uint8) + image_id-1

            y = resp[:,image_id-1, r,:]
            i = i+1
            image_bin.extend([x])
            image_ids.extend([image_vec])
            trial_ids.extend([trial_vec])

            spikes.extend([y])
            #print(index)
            #print(ms)
        #print(index)
        #print(x.shape)
        #print(x)
        #print(y.shape)
        #print(y)
    stim,spikes =  ( np.concatenate( np.array(image_bin) ),np.concatenate(np.array(spikes), axis=1).swapaxes(0,1))
    ids, trial = (np.concatenate(np.array(image_ids)),np.concatenate(np.array(trial_ids)))
    return (stim,spikes,ids,trial)

if __name__ == '__main__':
    main()
