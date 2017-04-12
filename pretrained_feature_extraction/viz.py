import joblib
import numpy as np
import plotly
from plotly.graph_objs import Histogram, Layout, Box, Figure
from plotly import tools
import plotly.figure_factory as ff

num_layers = 14
kf = 5

def load_data(early_filepath, late_filepath):
    early = joblib.load(early_filepath)
    late = joblib.load(late_filepath)
    return [early, late]

def make_figure(plt_data, out_file):
    fig = tools.make_subplots(rows=num_layers, cols=2)

    for i in np.arange(2):
        for j in np.arange(num_layers):
            d = plt_data[i][j]
            fig.append_trace(d['trace'], j+1, i+1)

    for i in np.arange(2*num_layers):
        axis = 'xaxis%g'%(i+1)
        fig['layout'][axis].update(title='ppcc', range=[-0.2, 0.8])

    plotly.offline.plot(fig, filename=out_file)



def hist_plot(early_filepath, late_filepath, fname='hist_plot'):

    results = load_data(early_filepath, late_filepath)
    labels=['early', 'late']

    plt_data = [np.arange(num_layers).tolist(),np.arange(num_layers).tolist()]
    means = []

    for i in np.arange(2):
        for j in np.arange(num_layers):
            d = results[i][j]
            data=np.array(d['ppcc_list'])
            network=d['network'][0]
            name=labels[i]+'_'+network

            plt_data[i][j] = dict(
                network=network,
                raw_data=data,
                mean=data.mean(),
                name=labels[i]+'_'+network,
                title='Mean: %.2f' % data.mean(),
                trace = Histogram(
                        x=data.flatten(),
                        opacity=0.75,
                        name=name)

                    )
    make_figure(plt_data, fname+'.html')

def line_plot(earlyfb, latefb):
    results = load_data(earlyfp, latefp)
    early, late = results[0], results[1]


def box_plot(earlyfp, latefp, fname='box_plot'):
    results = load_data(earlyfp, latefp)
    early, late = results[0], results[1]

    early_plt_data = []
    early_labels = []

    late_plt_data = []
    late_labels = []
    categories = []

    j=-1
    for e,l in zip(early, late):
        j=j+1
        data = [np.array(e['ppcc_list']).flatten(), np.array(l['ppcc_list']).flatten()]
        labels=[[e['network'][0] for _ in np.arange(kf*37)],[l['network'][0] for _ in np.arange(kf*37)]]
        early_plt_data.extend(data[0].tolist())
        late_plt_data.extend(data[1].tolist())

        early_labels.extend(labels[0])
        late_labels.extend(labels[1])

        network=e['network'][0]

    traces = [
        Box(
            y=early_plt_data,
            x=early_labels,
            name='early',
            # jitter=0.3,
            # pointpos= -1.8,
            # boxpoints = 'all',
            boxmean=True
        ),Box(
            y=late_plt_data,
            x=late_labels,
            name='late',
            # jitter=0.3,
            # pointpos= 1.8,
            # boxpoints='all',
            boxmean=True
        )]
    layout = Layout(
        yaxis=dict(
            title='ppcc',
            #zeroline=False,
            range=[-0.3, 0.8]
        ),
        boxmode='group'
    )
    fig = Figure(data=traces, layout=layout)
    plotly.offline.plot(fig, filename=fname+'.html')


def dist_plot(early_filepath, late_filepath):
    early = joblib.load(early_filepath)
    late = joblib.load(late_filepath)
    results = [early, late]
    label=['early', 'late']

    plt_data = []
    layouts = []
    figs = []
    for e,l in zip(early, late):
        data = [np.array(e['ppcc_list']).flatten(), np.array(l['ppcc_list']).flatten()]
        labels = ['early_'+e['network'][0], 'late_'+l['network'][0]]
        # means = data.mean(axis=0)
        fig = ff.create_distplot(data, labels, bin_size=0.1)
        fig['layout']['xaxis'].update(title='ppcc', range=[-0.2, 0.8])
        fig['layout']['yaxis'].update(title='#')
        figs.extend([fig])
        # trace = fig['data']
        # plt_data.extend([trace])
        # layouts.extend([fig['layout']])



    for i,f in enumerate(figs):
        plotly.offline.plot(f, filename='%g_traces.html'%i)


#box_plot('tmp/early_all_layers.pkl', 'tmp/late_all_layers.pkl')

hist_plot('tmp/early_sm_all_layers.pkl', 'tmp/late_sm_all_layers.pkl', 'sm_nat_ppcc_hist_layers_early_v_late')
box_plot('tmp/early_sm_all_layers.pkl', 'tmp/late_sm_all_layers.pkl', 'sm_nat_ppcc_box_layers_early_v_late')
