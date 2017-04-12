import scipy.io as sio
import numpy as np

def gen_y_fake(y, sem_y):
    loc = np.zeros_like(y)
    z = np.random.normal(loc,sem_y)
    return (y + z)

def pairwise_pcc(y,y_pred):
    # Expects data in shape [nsamples, ncells]

    ncells = y.shape[1]
    ppcc = [ np.corrcoef(y_pred[:,i],y[:,i]) for i in np.arange(ncells)]
    return np.nan_to_num(np.array(ppcc)[:,1,0])

def ppcc_max(activity_mean, activity_sem):
    # Expects shape [nsamples, ncells]

    y_fake = gen_y_fake(activity_mean, activity_sem)
    return pairwise_pcc(y_fake, activity_mean)


if __name__ == '__main__':
    mat_file = '../data/02_stats.mat'
    print('loading mat data...', mat_file)
    activity_contents = sio.loadmat(mat_file)
    activity = activity_contents['resp_mean'].swapaxes(0,1)
    activity_sem = activity_contents['resp_sem'].swapaxes(0,1)

    # Natural Images
    idxs = np.arange(540)[::2]

    ppcc_max = ppcc_max(activity[idxs], activity_sem[idxs])

    print(ppcc_max)

