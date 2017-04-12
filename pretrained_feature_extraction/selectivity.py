import scipy.io as sio
import numpy as np
import csv

def si(activity):
    # activity [956, n]
    nb_imgs = activity.shape[0]
    nb_neurons = activity.shape[1]
    si = []
    for n in np.arange(nb_neurons):
        sfr = [ np.power(fr,2)/nb_imgs for fr in activity[:,n]]
        sfr = np.array(sfr)
        num = np.power((activity[:,n]/nb_imgs).sum(),2)
        sp = (1 - num/sfr.sum())*(1-(1/nb_imgs))**-1
        si.extend([ sp ])
    return np.array(si)

if __name__ == '__main__':
    mat_content = sio.loadmat('../data/02_stats.mat')
    activity = mat_content['resp_mean'].T
    idxs = np.arange(540)[::2]
    si = si(activity)

    dir = '/home/elijahc/Dropbox/Kohn_Monkey_Data/v1_predictor_paper/data'
    with open(dir+'/02_selectivity_index.csv', 'w') as csvfile:
        fieldnames = ['cell','si']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i,si in enumerate(si):
            writer.writerow({'cell':i, 'si':si})
