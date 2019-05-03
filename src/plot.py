import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl
import matplotlib.cm as cm

# dirs = ['0201', '0401', '0301', '0601', '0701', '0801', '0901', '1101', '1201', '1301', '1401']
df = pd.concat([pd.read_pickle('../log/cluster/0305/*-v0.pkl')], ignore_index=True)

def quant_inf(x):
    return x.quantile(0.25)
def quant_sup(x):
    return x.quantile(0.75)

bmap = brewer2mpl.get_map('set1', 'qualitative', 9)
colors = bmap.mpl_colors

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# ys = ['agent', 'passenger', 'taxi']
# ys = ['light', 'sound', 'toy1', 'toy2']
print(df.columns)
# y = ['imitloss']
params = [
          '--log_dir',
          '--initq',
          '--layers',
          '--her',
          '--nstep',
          '--alpha',
          '--IS',
          '--targetClip',
          '--lambda',
          '--nbObjects',
          '--nbFeatures',
          '--nbActions',
          '--density',
          '--amplitude',
          '--evaluator',
          '--objects',
          '--exp4gamma',
          # '--exp4beta',
          '--exp4eta',
          '--goals',
          '--actions',
    '--dropout',
    '--l2reg',
    '--episodes',
    '--seed',
    '--experts'
          # '--seed'
          ]

df1 = df.copy()
df1 = df1[(df1['--env'] == 'Objects-v0')]
df1 = df1[(df1['--nstep'] == 1)]
df1 = df1[(df1['--dropout'] == 1)]
df1 = df1[(df1['--l2reg'] == 0)]
# df1 = df1[(df1['--exp4gamma'] == 0.01)]
# df1 = df1[(df1['--exp4beta'] == 100)]
# df1 = df1[(df1['--exp4eta'] == 0.01)]
#

# df1 = df1[(df1['--objectselector'] == 'exp4object')]

df1.fillna(method='ffill', inplace=True)

# df1 = df1[(df1['--objectselector'] == 'rndobject')]
y = ['train_reward_eval_eval']
x = ['trainstep']

paramsStudied = []
for param in params:
    l = df1[param].unique()
    print(param, l)
    if len(l) > 1:
        paramsStudied.append(param)
print(df1['num_run'].unique())

op_dict = {a:[np.median, np.mean, np.std, quant_inf, quant_sup] for a in y}
avg = 1
if avg:
    df1 = df1.groupby(x + params).agg(op_dict).reset_index()

print(paramsStudied)
a, b = 1,1

fig2, ax2 = plt.subplots(a, b, figsize=(18,10), squeeze=False, sharey=False, sharex=False)
p = 'num_run'
if avg:
    p = paramsStudied

for j, (name, g) in enumerate(df1.groupby(p)):
    if avg:

        if isinstance(name, tuple):
            label = ','.join(['{}:{}'.format(paramsStudied[k][2:], name[k]) for k in range(len(paramsStudied))])
        else:
            label = '{}:{}'.format(paramsStudied[0][2:], name)

    for i, valy in enumerate(y):
        xplot, yplot = i % a, i // a
        # print(g[valy])
        # ax2[xplot,yplot].plot(g['trainstep'], g[valy].rolling(5).mean(), label=valy, c=cm.hot(i/18))
        ax2[xplot,yplot].plot(g['trainstep'], g[valy]['median'], label=name, c=cm.hot(j/8))
        ax2[xplot,yplot].fill_between(g['trainstep'],
                                g[valy]['quant_inf'],
                                g[valy]['quant_sup'], alpha=0.25, linewidth=0, color=cm.hot(j/8))
        # ax2[i % a, i // a].scatter(g['train_step'], g[valy], s=1)
        ax2[xplot, yplot].set_title(label=valy)
        # if i == 0: ax2[xplot, yplot].legend()
        # ax2[xplot, yplot].set_xlim([0, 100000])
        # ax2[xplot, yplot].set_ylim([0, 0.01])
    ax2[xplot,yplot].legend()

plt.show()
