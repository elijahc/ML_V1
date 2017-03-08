import scipy.io as sio
import numpy as np
import csv

def si(activity):
    # activity [956, n]
    nb_imgs = activity.shape[0]
    nb_neurons = activity.shape[1]
    si = []
    for n in np.arange(nb_neurons):
        sfr = [ np.power(fr,2) for fr in activity[:,n]]
        sfr = np.array(sfr)
        denom = np.power(activity[:,n].sum(),2)
        si.extend([ (sfr.sum()/denom)])
    return np.array(si)

if __name__ == '__main__':
    mat_content = sio.loadmat('../data/02mean_d1.mat')
    activity = mat_content['activity']
    si = si(activity)

    with open('selectivity_all.csv', 'w') as csvfile:
        fieldnames = ['neuron','si']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i,si in enumerate(si):
            writer.writerow({'neuron':i, 'si':si})
