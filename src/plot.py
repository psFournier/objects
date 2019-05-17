import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl
import matplotlib.cm as cm

# dirs = ['0201', '0401', '0301', '0601', '0701', '0801', '0901', '1101', '1201', '1301', '1401']
df = pd.concat([pd.read_pickle('../log/cluster/1705/*.pkl')], ignore_index=True)

def quant_inf(x):
    return x.quantile(0.2)
def quant_sup(x):
    return x.quantile(0.8)

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
          # '--nbFeatures',
          # '--nbActions',
          # '--density',
          # '--amplitude',
          '--evaluator',
          '--objects',
          '--exp4gamma',
          # '--exp4beta',
          '--exp4eta',
          '--goals',
          '--actions',
    '--dropout',
    '--l2reg',
    # '--episodes',
    # '--seed',
    '--experts',
    '--nbObjectsTrain'
]

df1 = df.copy()
# df1 = df1[(df1['--env'] == 'ObjectsPlayroom')]
df1 = df1[(df1['--nstep'] == 1)]
df1 = df1[(df1['--objects'] == 'uni')]
df1 = df1[(df1['--l2reg'] == 0)]
# df1 = df1[(df1['--actions'] == 'sgsoft')]
# df1 = df1[(df1['--nbObjectsTrain'] == 1)]
# df1 = df1[(df1['--exp4gamma'] == 0.01)]
df1 = df1[(df1['--her'] == 0)]
df1 = df1[(df1['--initq'] == -100)]
#

# df1 = df1[(df1['--objectselector'] == 'exp4object')]

# df1.fillna(value=0, inplace=True)

# df1 = df1[(df1['--objectselector'] == 'rndobject')]
expert_probs = ['lp_0.001_0.1_50_expert_probs',
                 'lp_0.001_1_50_expert_probs', 'lp_0.001_5_50_expert_probs',
                 'lp_0.01_0.1_50_expert_probs', 'lp_0.01_1_50_expert_probs',
                 'lp_0.01_5_50_expert_probs', 'lp_0.1_0.1_50_expert_probs',
                 'lp_0.1_1_50_expert_probs', 'lp_0.1_5_50_expert_probs']
y = [e+'_{}'.format(i) for e in expert_probs for i in range(9)]
x = ['trainstep']

paramsStudied = []
for param in params:
    l = df1[param].unique()
    print(param, l)
    if len(l) > 1:
        paramsStudied.append(param)
print(df1['num_run'].unique())

array_columns = ['dqn_model_qvals', 'dqn_model_rewards',
                 'dqn_model_targets',
                 'dqn_model_tderrors', 'envsteps',
                 'lp_0.001_0.1_50_expert_probs',
                 'lp_0.001_1_50_expert_probs', 'lp_0.001_5_50_expert_probs',
                 'lp_0.01_0.1_50_expert_probs', 'lp_0.01_1_50_expert_probs',
                 'lp_0.01_5_50_expert_probs', 'lp_0.1_0.1_50_expert_probs',
                 'lp_0.1_1_50_expert_probs', 'lp_0.1_5_50_expert_probs', 'test_ep_eval_rewards',
                 'test_ep_eval_tderrors',
                 'train_ep_eval_rewards', 'train_ep_eval_tderrors','trainsteps','player_rewards', 'player_tderrors']

def array_to_vals(x):
    return [xi for xi in x['__ndarray__']]

for column in array_columns:
    temp = list(zip(*df1[column].map(array_to_vals)))
    for i, l in enumerate(temp):
        df1[column+'_{}'.format(i)] = l
    df1.drop(column, axis=1)

op_dict = {a:[np.median, np.mean, np.std, quant_inf, quant_sup] for a in y}
avg = 1
if avg:
    df1 = df1.groupby(x + params).agg(op_dict).reset_index()

print(paramsStudied)
a, b = 3,3

fig2, ax2 = plt.subplots(a, b, figsize=(18,10), squeeze=False, sharey=False, sharex=True)
p = 'num_run'
if avg:
    if not paramsStudied:
        paramsStudied.append('--her')
    p = paramsStudied

for j, (name, g) in enumerate(df1.groupby(p)):
    if avg:

        if isinstance(name, tuple):
            label = ','.join(['{}:{}'.format(paramsStudied[k][2:], name[k]) for k in range(len(paramsStudied))])
        else:
            label = '{}:{}'.format(paramsStudied[0][2:], name)

    for i, valy in enumerate(y):
        k = i//9
        xplot, yplot = k % a, k // a
        # xplot, yplot = 0, 0
        # print(g[valy])
        # print(g['trainstep'])
        # ax2[xplot,yplot].plot(g['trainstep'], g[valy], label=valy, c=cm.hot(i/18))
        ax2[xplot,yplot].plot(g['trainstep'], g[valy]['median'].rolling(5).mean(), label=i%9, c=colors[i%9])
        ax2[xplot,yplot].fill_between(g['trainstep'],
                                g[valy]['quant_inf'].rolling(5).mean(),
                                g[valy]['quant_sup'].rolling(5).mean(), alpha=0.25, linewidth=0, color=colors[i%9])
        # ax2[i % a, i // a].scatter(g['train_step'], g[valy], s=1)
        ax2[xplot, yplot].set_title(label=expert_probs[k])
        # if i == 0: ax2[xplot, yplot].legend()
        # ax2[xplot, yplot].set_xlim([0, 25000])
        # ax2[xplot, yplot].set_ylim([0, 0.01])
    ax2[xplot,yplot].legend()

plt.show()
