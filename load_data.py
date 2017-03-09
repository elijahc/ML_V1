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
                        help='Structure to parse the data into default: ledger')
    parser.add_argument('--format', type=str, default='mat', choices=['mat','hdf5','csv','pickle'],
                        help='File Format to save data default: mat')

    FLAGS = parser.parse_args()
    # Load files
    print('loading...stim_sequence')
    stim_sequence = sio.loadmat('data/stimulus_sequence.mat')['stimulus_sequence']
    FILE = FLAGS.infile

    print('loading...', FILE)
    mat_file = sio.loadmat(FILE)

    #%%
    # Filter out poor quality neurons
    mask = np.squeeze(mat_file['INDCENT']).astype(bool)

    resp_train = mat_file['resp_train'][mask]
    stim_len = np.size(resp_train,axis=-1)
    resp_train_blk = mat_file['resp_train_blk'][mask]
    blank_len = np.size(resp_train_blk,axis=-1)

    # Shift by 50ms to account for response latency
    latency = 50
    resp = np.concatenate((resp_train,resp_train_blk), axis=3)
    resp = np.roll(resp,-latency,3)[:,:,:,:-latency]


    resp_mean, resp_std, resp_sem = trial_stats(resp)
    resp_nat_sm, resp_nat_lg = subdivide(resp[:,:,:,:50])
    stim, spike_train,ids,trial = mutate(resp,stim_len,blank_len,stim_sequence)
    import pdb; pdb.set_trace()
    out_dict = dict(
            timeseries=spike_train,
            resp_mean=resp_mean,
            resp_std=resp_std,
            resp_sem=resp_sem,
            #nat_resp_sm=resp_nat_sm,
            #nat_resp_lg=resp_nat_lg,
            stim=stim,
            trial_num=trial,
            image_id=ids)

    outfile = FLAGS.outfile
    print('writing ', outfile, '...')
    sio.savemat(outfile, out_dict)

def trial_stats(resp):
    t_win = np.size(resp, 3)
    resp = resp.sum(axis=3)
    resp_mean = resp.mean(axis=2)
    resp_std = resp.std(axis=2)
    resp_sem = resp_std/np.sqrt(20)
    return (resp_mean, resp_std, resp_sem)

def subdivide(resp):
    tmp = np.squeeze(resp[:,:(2*9*30),:])
    tmp = tmp.reshape(np.size(resp,0),2,9,30,20,np.size(resp,-1))
    resp_nat_sm = tmp[:,0,:,:,:].reshape(np.size(tmp,0),(9*30),20,np.size(tmp,-1))
    resp_nat_lg = tmp[:,1,:,:,:].reshape(np.size(tmp,0),(9*30),20,np.size(tmp,-1))
    return (resp_nat_sm, resp_nat_lg)

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
