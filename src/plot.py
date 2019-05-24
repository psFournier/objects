import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl
import matplotlib.cm as cm

# dirs = ['0201', '0401', '0301', '0601', '0701', '0801', '0901', '1101', '1201', '1301', '1401']
df = pd.concat([
    # pd.read_pickle('../log/cluster/2205bis/*.pkl'),
    pd.read_pickle('../log/cluster/2205/*.pkl'),
    # pd.read_pickle('../log/cluster/2305/Objects4-v0.pkl'),
], ignore_index=True)

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
          # '--log_dir',
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
    '--nbObjectsTrain',
    '--agentEta'
]

df1 = df.copy()
df1 = df1[(df1['--env'] == 'Objects4-v0')]
# df1 = df1[(df1['--nstep'] == 1)]
# df1 = df1[((df1['--objects'] == 'uni') & (df1['--exp4gamma'] == 0.3))|(df1['--objects'] == 'exp4')]
# df1 = df1[(df1['--objects'] == 'exp4')]
df1 = df1[(df1['--nbObjects'] == 1)]
# df1 = df1[(df1['--nbObjectsTrain'] == 1)]
# df1 = df1[(df1['--exp4gamma'] == 0.3)]
# df1 = df1[(df1['--agentEta'] == 0.01)]
df1 = df1[(df1['--layers'] == '8,8')]
#

# df1 = df1[(df1['--objectselector'] == 'exp4object')]

# df1.fillna(value=0, inplace=True)

# df1 = df1[(df1['--objectselector'] == 'rndobject')]
# experts = ['lp_0.001_0.1_50_expert',
#                  'lp_0.001_1_50_expert', 'lp_0.001_5_50_expert',
#                  'lp_0.01_0.1_50_expert', 'lp_0.01_1_50_expert',
#                  'lp_0.01_5_50_expert', 'lp_0.1_0.1_50_expert',
#                  'lp_0.1_1_50_expert', 'lp_0.1_5_50_expert']
experts = ['obj_'+str(i)+'_expert' for i in range(2)]
# y = [e+'_probs_{}'.format(i) for e in expert_probs for i in range(9)]
# y = ['exp4_obj_weight_'+e for e in expert_probs]
x = ['envstep']
y = ['dqn_model_tderrors_0', 'dqn_model_tderrors_1',
     'trainsteps_0', 'trainstep'
     ]
# y = ['player_rewards_8']

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
                 'test_ep_eval_rewards',
                 'train_ep_eval_rewards', 'trainsteps','player_rewards']
array_columns += [e+'_probs' for e in experts]

def array_to_vals(x):
    res = []
    if np.isscalar(x):
        res.append(x)
    else:
        res+=list(x['__ndarray__'][::-1])
    return res

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
a, b = 1,2

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

    for i, valy in enumerate(y[:-2]):
        # k = j//2
        # xplot, yplot = k % a, k // a
        xplot, yplot = i % a, i // a
        # print(g[valy])
        # print(g['num_run'])
        # print(g['trainstep'])
        # print(g[valy])
        # ax2[xplot,yplot].plot(g['trainstep'], g[valy], label=name, c=colors[j%9])
        # l = valy
        l = j%4
        ax2[xplot,yplot].plot(g['trainstep']['mean'], g[valy]['median'].rolling(5).mean(), label=name, c=colors[j%9])
        ax2[xplot,yplot].fill_between(g['trainstep']['mean'],
                                g[valy]['quant_inf'].rolling(5).mean(),
                                g[valy]['quant_sup'].rolling(5).mean(), alpha=0.25, linewidth=0, color=colors[j%9])
        # ax2[i % a, i // a].scatter(g['train_step'], g[valy], s=1)
        # title = experts[i]
        ax2[xplot, yplot].set_title(label=valy)
        # if i == 0: ax2[xplot, yplot].legend()
        # ax2[xplot, yplot].set_xlim([0, 25000])
        # ax2[xplot, yplot].set_ylim([0, 0.01])
        ax2[xplot,yplot].legend()

plt.show()
