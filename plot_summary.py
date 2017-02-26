import numpy as np
import pandas as pd
import scipy.io as sio
from bokeh.palettes import Viridis5 as palette
from bokeh.plotting import figure, show, output_file
from bokeh.charts import Scatter,HeatMap
from bokeh.charts.attributes import *

import pickle

df = pd.read_pickle('tmp/02_timeseries_long.pkl')

output_file('./scatter3.html')
p = Scatter(df,
            x='time',
            y='nid',
            color=color('spikes',palette=palette),
            webgl=True,
            background_fill_alpha=0.3,
            legend='top_right',
            width=1500,
            height=750,
            marker='square')

show(p)
