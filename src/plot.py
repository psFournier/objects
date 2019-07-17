import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib import rc, rcParams
from matplotlib import lines
rc('font',family='serif')
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"
                                 r"\usepackage{amssymb}"]

df = pd.concat([
    pd.read_pickle('../log/local/Objects-v0.pkl'),
    # pd.read_pickle('../log/local/ObjectsForGeneralization-v0/*.pkl'),
    # pd.read_pickle('../log/cluster/2305/Objects4-v0.pkl'),
], ignore_index=True)


def quant_inf(x):
    return x.quantile(0.2)


def quant_sup(x):
    return x.quantile(0.8)


bmap = brewer2mpl.get_map('set1', 'qualitative', 9)
colors = bmap.mpl_colors


def get_cmap(n, name='hsv'):
    # Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    # RGB color; the keyword argument name must be a standard mpl colormap
    #  name
    return plt.cm.get_cmap(name, n)


params = [
    '--initq',
    '--layers',
    '--her',
    '--nstep',
    '--alpha',
    '--IS',
    '--lambda',
    '--evaluator',
    '--objects',
    '--expgamma',
    # '--exp4beta',
    '--expeta',
    '--goals',
    '--actions',
    '--dropout',
    '--l2reg',
    # '--episodes',
    # '--seed',
    '--experts',
    '--agentEta',
    '--globalEval',
    '--episodeSteps',
    '--env'
]

df1 = df.copy()
# df1 = df1[(df1['--env'] == 'Objects-v0')]
# df1 = df1[(df1['--nstep'] == 1)]
# df1 = df1[((df1['--objects'] == 'uni') & (df1['--exp4gamma'] == 0.3)) |
#           (df1['--objects'] == 'exp4')]
# df1 = df1[(df1['--objects'] == 'uni')]
# df1 = df1[(df1['--nbObjects'] == 50)]
# df1 = df1[(df1['--expgamma'] == 1)]
# df1 = df1[(df1['--layers'] == '64,64')]
# df1 = df1[(df1['--initq'] == 0)]
# df1 = df1[(df1['--her'] == 4)]
#
#
# df1 = df1[(df1['--objectselector'] == 'exp4object')]
#
# df1.fillna(value=0, inplace=True)
#
# df1 = df1[(df1['--objectselector'] == 'rndobject')]
# experts = ['lp_0.001_0.1_50_expert',
#                  'lp_0.001_1_50_expert', 'lp_0.001_5_50_expert',
#                  'lp_0.01_0.1_50_expert', 'lp_0.01_1_50_expert',
#                  'lp_0.01_5_50_expert', 'lp_0.1_0.1_50_expert',
#                  'lp_0.1_1_50_expert', 'lp_0.1_5_50_expert']
# experts = ['obj_'+str(i)+'_expert' for i in range(2)]
# y = [e+'_probs_{}'.format(i) for e in expert_probs for i in range(9)]
# y = ['exp4_obj_weight_'+e for e in expert_probs]

x = ['envstep']
y = ['player_rewards_0',
     'test_generalization_rewards_0',
     'test_generalization_rewards_1',
     'trainsteps']

paramsStudied = []
for param in params:
    l = df1[param].unique()
    print(param, l)
    if len(l) > 1:
        paramsStudied.append(param)
print(df1['num_run'].unique())

array_columns = ['trainsteps',
                 'player_rewards',
                 'test_generalization_rewards']

def array_to_vals(x):
    # Take the numpy array in data frames logged, and converts them in as many
    # separate columns as necessary, plus the mean of the columns
    res = []
    if np.isscalar(x):
        res.append(x)
    else:
        res += list(x['__ndarray__'][:])
        res.append(np.mean(x['__ndarray__'][:]))
    return res


for column in array_columns:
    temp = list(zip(*df1[column].map(array_to_vals)))
    for i, l in enumerate(temp[:-1]):
        df1[column + '_{}'.format(i)] = l
    df1[column] = temp[-1]
print(df1.columns)

op_dict = {a: [np.median, np.mean, np.std, quant_inf, quant_sup] for a in y}
avg = 1
if avg:
    df1 = df1.groupby(x + params).agg(op_dict).reset_index()
print(paramsStudied)
a, b = 1,1

fig, axes = plt.subplots(a, b,
                         figsize=(15,9),
                         squeeze=False,
                         sharey=True,
                         sharex=True)
p = 'num_run'
if avg:
    if not paramsStudied:
        paramsStudied.append('--her')
    p = paramsStudied

styles = [{'c': colors[j], 'linestyle': '-', 'linewidth': 3} for j in range(3)]

for j, (name, g) in enumerate(df1.groupby(p)):
    if avg:

        if isinstance(name, tuple):
            label = ','.join(
                ['{}:{}'.format(paramsStudied[k][2:], name[k]) for k in
                 range(len(paramsStudied))])
        else:
            label = '{}:{}'.format(paramsStudied[0][2:], name)

    for i, valy in enumerate(y[:-1]):
        xplot, yplot = j % a, j // a
        trainmean = g['trainsteps']['mean']
        valmed = g[valy]['median']
        axes[xplot, yplot].plot(trainmean,
                               valmed.rolling(5).mean(),
                               # label=valy,
                               **styles[i])
        axes[xplot, yplot].fill_between(trainmean,
                                       g[valy]['quant_inf'].rolling(5).mean(),
                                       g[valy]['quant_sup'].rolling(5).mean(),
                                       alpha=0.25,
                                       linewidth=0,
                                       color=styles[j]['c'])

        # ax2[xplot, yplot].set_title(label=valy)
        # ax2[xplot, yplot].legend()
        # ax2[xplot, yplot].set_xlim([0, 300000])
        # ax2[xplot, yplot].set_ylim([0, 0.01])

        # valmean = g[valy]['mean']
        # valstdminus = valmean - 0.5 * g[valy]['std']
        # valstdplus = valmean + 0.5 * g[valy]['std']
        # ax2[xplot, yplot].fill_between(trainmean,
        #                                valstdminus.rolling(5).mean(),
        #                                valstdplus.rolling(5).mean(),
        #                                alpha=0.25,
        #                                linewidth=0,
        #                                color=colors[j % 9])
        # ax2[i % a, i // a].scatter(g['train_step'], g[valy], s=1)
        # title = experts[i]
        # if i == 0: ax2[xplot, yplot].legend()

fontsize = 24
handles0 = [lines.Line2D([], [], label=l, **styles[i])
            for i,l in enumerate([r'Training object',
                                  r'Similar test objects',
                                  r'Dissimilar test objects'
                                  ])]
axes[0, 0].legend(loc='upper left',
                  bbox_to_anchor=(0, 1),
                  title='Episode mean return for:',
                  handles=handles0,
                  fontsize=fontsize,
                  framealpha=1)
axes[0, 0].legend_.get_title().set_fontsize(fontsize)

axes[0, 0].set_xlim([0, 100000])
axes[0, 0].set_xlabel(xlabel=r'\textbf{Training steps}',
                      fontsize=fontsize)
axes[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(25))
# axes[0, 0].text(x=0.01, y=0.9, s=r'\textbf{A}', fontsize=fontsize, transform=axes[0, 0].transAxes)
axes[0, 0].set_ylabel(ylabel=r'$\mathbf{R}$', fontsize=fontsize)
axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(10000))
axes[0, 0].ticklabel_format(style='sci', scilimits=(-3,4), axis='both')
axes[0, 0].xaxis.get_offset_text().set_fontsize(fontsize)
for tick in axes[0, 0].xaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
for tick in axes[0, 0].yaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
fig.tight_layout(h_pad=2)



plt.savefig("/home/pierre/Latex/manuscrit/images/objects/result1.jpeg",
            bbox_inches='tight')

plt.show()
