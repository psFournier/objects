import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl

dirs = ['0201', '0401', '0301', '0601', '0701', '0801', '0901', '1101', '1201', '1301', '1401']
# dirs = ['1401']
df = pd.concat([pd.read_pickle('../log/local/*-v0.pkl')], ignore_index=True)

def quant_inf(x):
    return x.quantile(0.25)
def quant_sup(x):
    return x.quantile(0.75)

bmap = brewer2mpl.get_map('set1', 'qualitative', 9)
colors = bmap.mpl_colors

# ys = ['agent', 'passenger', 'taxi']
# ys = ['light', 'sound', 'toy1', 'toy2']
print(df.columns)
# y = ['imitloss']
x = ['step']
params = [
          '--log_dir',
          '--initq',
          '--layers',
          '--her',
          '--nstep',
          '--alpha',
          '--IS',
          '--targetClip',
          '--lambda'
          ]

a, b = 1,1
fig, axes = plt.subplots(a, b, figsize=(15,12), squeeze=False, sharey=True, sharex=True)

df1 = df.copy()
df1 = df1[(df1['--env'] == 'Objects-v0')]
# df1 = df1[(df1['--demoorder'] == 0)]
y1 = ['C{}'.format(str(s)) for s in [i for i in [2, 3, 4, 5, 6, 7]]]
df1['C'] = df1[y1].apply(np.mean, axis=1)
p1 = [p for p in params if len(df1[p].unique()) > 1]
for p in params: print(p, df1[p].unique())
df1 = df1.groupby(x + params).agg({'C':[np.median, np.mean, np.std, quant_inf, quant_sup]}).reset_index()

for j, (name, g) in enumerate(df1.groupby(p1)):
    axes[0, 0].plot(g['step'], g['C']['median'], c=colors[j%9], label=name)
    axes[0, 0].fill_between(g['step'],
                           g['C']['quant_inf'],
                           g['C']['quant_sup'], alpha=0.25, linewidth=0, color=colors[j%9])
    axes[0,0].legend()
    # axes[0, 0].set_xlim([0, 800000])
    # ax2[0, 0].set_ylim([0, 1])

    # axes[0, 0].set_xlabel('Environment step', fontsize='large')
    axes[0, 0].set_ylabel('Average control', fontsize='large')

# plt.savefig("/home/pierre/Latex/Papiers_blogs/ICML2019/16199341sknbhtkjwgjg/result9.jpeg", bbox_inches='tight')

plt.show()