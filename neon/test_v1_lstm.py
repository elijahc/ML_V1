import scipy.io as sio
from v1_lstm import TimeSeries, DataIteratorSequence, FractionExplainedVariance
from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.layers import GeneralizedCost, LSTM, Affine, RecurrentLast
from neon.models import Model
from neon.optimizers import Adam, ExpSchedule
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
hidden = 32

be = gen_backend(**extract_valid_args(args, gen_backend))

train_set = DataIteratorSequence(ts.train, seq_len, forward=3, return_sequences=True)
valid_set = DataIteratorSequence(ts.test, seq_len, forward=3, return_sequences=True)

init = GlorotUniform()

layers = [
    LSTM(hidden, init, activation=Logistic(),
        gate_activation=Tanh(), reset_cells=False),
    Affine(train_set.nfeatures, init, bias=init, activation=Identity())
        ]

model = Model(layers=layers)

#s = ExpSchedule(decay=0.7)
cost = GeneralizedCost(SumSquared())

#optimizer = RMSProp(stochastic_round=args.rounding)
optimizer = Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999)

callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)
callbacks.add_callback(MetricCallback(eval_set=valid_set, metric=FractionExplainedVariance(), epoch_freq=args.eval_freq))
#callbacks.add_callback(MetricCallback(eval_set=valid_set,metric=Accuracy(),  epoch_freq=args.eval_freq))

model.fit(train_set,
          optimizer=optimizer,
          num_epochs=args.epochs,
          cost=cost,
          callbacks=callbacks)


