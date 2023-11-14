import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 75


labels = ['Layer insertion', 'FNN1']

with open(f"results_data/Exp{k}_1.json") as file:
    # a_temp = list(json.load(file).values())
    f = json.load(file)
    a_temp = [f[i]['losses'] for i in f.keys()]
    a = np.array(a_temp)
    
    print(f'shape of a {a.shape}')

if True:
    with open(f"results_data/Exp{k}_2.json") as file: #Exp8_3
        # b_temp = list(json.load(file).values())
        f = json.load(file)
        b_temp = [f[i]['losses'] for i in f.keys()]
        
        b = np.array(b_temp)
        




matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('xtick', labelsize=16)
plt.figure(figsize=(20,5))

plt.plot(a[0,:], '-', label=labels[0], linewidth=3)
plt.plot(b[0,:], '--', label=labels[1], linewidth=3)
plt.vlines(80,min(a[0,:]),max(a[0,:]),linestyles='dotted',colors='gray')

plt.yscale('log')
plt.xlabel('iterations', fontsize=20)
plt.ylabel('loss', fontsize=20)
#plt.ylim()
#plt.xlim(left=70, right=140)
plt.legend(fontsize=20)
#plt.title(f'{i}')


plt.tight_layout()
plt.savefig('tikzpicture_plots/fig3_aus75.pdf', format="pdf", bbox_inches="tight")
plt.show()