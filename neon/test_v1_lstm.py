import scipy.io as sio
from v1_lstm import TimeSeries
from v1_lstm import DataIteratorSequence
from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.layers import GeneralizedCost, LSTM, Affine, RecurrentLast
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Logistic, Tanh, Identity, MeanSquared
from neon.callbacks.callbacks import Callbacks
from neon import logger as neon_logger
from neon.util.argparser import NeonArgparser, extract_valid_args
from sklearn.preprocessing import MinMaxScaler

parser = NeonArgparser(__doc__)
args = parser.parse_args(gen_be=False)

mat_data = sio.loadmat('../data/timeseries/02_timeseries.mat')['timeseries']

scaler = MinMaxScaler(feature_range=(0,1))
input_data = scaler.fit_transform(mat_data)
ts = TimeSeries(input_data)

seq_len = 30
hidden = 32

be = gen_backend(**extract_valid_args(args, gen_backend))

train_set = DataIteratorSequence(ts.train, seq_len, return_sequences=True)
valid_set = DataIteratorSequence(ts.test, seq_len, return_sequences=True)
import pdb; pdb.set_trace()

init = GlorotUniform()

layers = [
    LSTM(hidden, init, activation=Logistic(),
        gate_activation=Tanh(), reset_cells=False),
    Affine(train_set.nfeatures, init, bias=init, activation=Identity())
        ]

model = Model(layers=layers)
cost = GeneralizedCost(MeanSquared())
optimizer = RMSProp(stochastic_round=args.rounding)

callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)

model.fit(train_set,
          optimizer=optimizer,
          num_epochs=args.epochs,
          cost=cost,
          callbacks=callbacks)


