import scipy.io as sio
import matplotlib.pyplot as plt
from v1_lstm import *
from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.layers import GeneralizedCost, LSTM, Affine, RecurrentLast, Dropout, GRU
from neon.models import Model
from neon.optimizers import Adam, ExpSchedule, RMSProp
from neon.transforms import *
from neon.transforms.cost import Metric
from neon.callbacks.callbacks import Callbacks, MetricCallback
from neon import logger as neon_logger
from neon.util.argparser import NeonArgparser, extract_valid_args

parser = NeonArgparser(__doc__)
args = parser.parse_args(gen_be=False)

mat_data = sio.loadmat('../data/timeseries/02_timeseries.mat')['timeseries']

ts = TimeSeries(mat_data)

seq_len = 40
hidden = 64

be = gen_backend(**extract_valid_args(args, gen_backend))

train_set = DataIteratorSequence(ts.train, seq_len, forward=3, return_sequences=True)
valid_set = DataIteratorSequence(ts.test, seq_len, forward=3, return_sequences=True)

init = GlorotUniform()

layers = [
    GRU(hidden, init, activation=Tanh(),
        gate_activation=Logistic(), reset_cells=False),
    Dropout(keep=0.5),
    Affine(train_set.nfeatures, init, bias=init, activation=Identity())
        ]

model = Model(layers=layers)

#s = ExpSchedule(decay=0.7)
weights = np.expand_dims(np.arange(seq_len),axis=1)
weights = np.repeat(weights, args.batch_size,axis=1)
weights = np.repeat(np.expand_dims(weights,axis=0),37,axis=0)
weights = np.reciprocal(weights.astype(np.float)+1).reshape(37,-1) * 10000

#cost = GeneralizedCost(SumSquared())
cost = GeneralizedCost(WeightedSumSquared(weights=weights))

#optimizer = RMSProp(stochastic_round=args.rounding)
optimizer = Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999)

callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)
#callbacks.add_callback(MetricCallback(eval_set=valid_set, metric=FractionExplainedVariance(), epoch_freq=args.eval_freq))
#callbacks.add_callback(MetricCallback(eval_set=valid_set,metric=Accuracy(),  epoch_freq=args.eval_freq))

model.fit(train_set,
          optimizer=optimizer,
          num_epochs=args.epochs,
          cost=cost,
          callbacks=callbacks)

train_output = model.get_outputs(
train_set).reshape(-1, train_set.nfeatures)
valid_output = model.get_outputs(
valid_set).reshape(-1, valid_set.nfeatures)
train_target = train_set.y_series
valid_target = valid_set.y_series

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


def fev(y,t):
    self.y = y
    self.y = t
