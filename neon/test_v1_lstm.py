import scipy.io as sio
import matplotlib.pyplot as plt
from v1_lstm import *
from costs_v1 import *
from layers_v1 import *
from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.layers import GeneralizedCost, LSTM, Affine, RecurrentLast, Dropout, GRU, RecurrentSum
from neon.models import Model
from neon.optimizers import *
from neon.transforms import *
from neon.transforms.cost import Metric
from neon.callbacks.callbacks import Callbacks, MetricCallback
from neon import logger as neon_logger
from neon.util.argparser import NeonArgparser, extract_valid_args

def main():
    parser = NeonArgparser(__doc__)
    args = parser.parse_args(gen_be=False)

    mat_data = sio.loadmat('../data/timeseries/02_timeseries.mat')

    ts = V1TimeSeries(mat_data['timeseries'], mat_data['stim'], binning=10)

    seq_len = 30
    hidden = 20

    be = gen_backend(**extract_valid_args(args, gen_backend))

    train_spike_set = V1IteratorSequence(ts.train, seq_len, return_sequences=False)
    valid_spike_set = V1IteratorSequence(ts.test, seq_len, return_sequences=False)
    import pdb; pdb.set_trace()

    init = GlorotUniform()

    # dataset = MNIST(path=args.data_dir)
    # (X_train, y_train), (X_test, y_test), nclass = dataset.load_data()
    # train_set = ArrayIterator([X_train, X_train], y_train, nclass=nclass, lshape=(1, 28, 28))
    # valid_set = ArrayIterator([X_test, X_test], y_test, nclass=nclass, lshape=(1, 28, 28))

    # # weight initialization
    # init_norm = Gaussian(loc=0.0, scale=0.01)

    # # initialize model
    # path1 = Sequential(layers=[Affine(nout=100, init=init_norm, activation=Rectlin()),
    #                            Affine(nout=100, init=init_norm, activation=Rectlin())])

    # path2 = Sequential(layers=[Affine(nout=100, init=init_norm, activation=Rectlin()),
    #                            Affine(nout=100, init=init_norm, activation=Rectlin())])

    # layers = [MergeMultistream(layers=[path1, path2], merge="stack"),
    #           Affine(nout=10, init=init_norm, activation=Logistic(shortcut=True))]

    spike_rnn_path = Sequential( layers = [

        LSTM(hidden, init, activation=Logistic(),
            gate_activation=Logistic(), reset_cells=False),

        Dropout(keep=0.5),

         LSTM(hidden, init, activation=Logistic(),
             gate_activation=Logistic(), reset_cells=False),

        #Dropout(keep=0.85),

        RecurrentLast(),

        Affine(train_set.nfeatures, init, bias=init, activation=Identity(), name='spike_in')])

    stim_rnn_path = Sequential( layers = [

        LSTM(hidden, init, activation=Logistic(),
            gate_activation=Logistic(), reset_cells=False),

        Dropout(keep=0.5),

        RecurrentLast(),
        Affine(1, init, bias=init, activation=Identity(), name='stim')])

    layers = [
            MergeMultiStream(
                layers = [
                    spike_rnn_path,
                    stim_rnn_path],
                merge="stack"),

            Affine(train_set.nfeatures, init, bias=init, activation=Identity(), name='spike_out'),

            Round()
            ]

    model = Model(layers=layers)

    sched = ExpSchedule(decay=0.7)

    # cost = GeneralizedCost(SumSquared())
    cost = GeneralizedCost(MeanSquared())

    optimizer_two = RMSProp(stochastic_round=args.rounding)
    optimizer_one = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9, schedule=sched)

    opt = MultiOptimizer({'default': optimizer_one,
                          'Bias': optimizer_two,
                          'special_linear': optimizer_two})

    callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)
    callbacks.add_hist_callback(filter_key = ['W'])
    #callbacks.add_callback(MetricCallback(eval_set=valid_set, metric=FractionExplainedVariance(), epoch_freq=args.eval_freq))
    #callbacks.add_callback(MetricCallback(eval_set=valid_set,metric=Accuracy(),  epoch_freq=args.eval_freq))

    model.fit(train_set,
              optimizer=opt,
              num_epochs=args.epochs,
              cost=cost,
              callbacks=callbacks)

    train_output = model.get_outputs(
    train_set).reshape(-1, train_set.nfeatures)
    valid_output = model.get_outputs(
    valid_set).reshape(-1, valid_set.nfeatures)
    train_target = train_set.y_series
    valid_target = valid_set.y_series

    tfev = fev(train_output, train_target, train_set.mean)
    vfev = fev(valid_output, valid_target, valid_set.mean)

    neon_logger.display('Train FEV: %g, Valid FEV:  %g' % (tfev, vfev))
    # neon_logger.display('Train Mean: %g, Valid Mean:  %g' % (train_set.mean, valid_set.mean))

    plt.figure()
    plt.plot(train_output[:, 0], train_output[
        :, 1], 'bo', label='prediction')
    plt.plot(train_target[:, 0], train_target[:, 1], 'r.', label='target')
    plt.legend()
    plt.title('Neon on training set')
    plt.savefig('neon_series_training_output.png')

    plt.figure()
    plt.plot(valid_output[:, 0], valid_output[
        :, 1], 'bo', label='prediction')
    plt.plot(valid_target[:, 0], valid_target[:, 1], 'r.', label='target')
    plt.legend()
    plt.title('Neon on validation set')
    plt.savefig('neon_series_validation_output.png')

def fev(y,t, x_mean):
    sse = np.square(y-t).sum()
    var = np.square(x_mean.round()-t).sum()
    fev = 1 - (sse/var)

    return fev

if __name__ == '__main__':
    main()
