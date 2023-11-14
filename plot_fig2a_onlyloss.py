import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 9
line = 70

std = False
plot_b = True # plot bigger net classical
plot_c = True # plot small net classical 
plot_error = True
plot_area = False
show_exit_flag = False
log_scale = True
smooth = 0. # not needed for GD
no_of_steps_included = 1000
if plot_b:
    labels = ['Layer insertion', 'FNN2', 'FNN1'] # a
    #labels = ['Layer insertion', 'ResNet1', 'ResNet2'] #b
if not plot_b:
    labels = ['layer insertion', 'FNN1']

with open(f"results_data/Exp{k}_1.json") as file:
    # a_temp = list(json.load(file).values())
    f = json.load(file)
    a_temp = [f[i]['losses'] for i in f.keys()]
    a_struct = [f[i]['structures'] for i in f.keys()]
    if plot_error:
        a_err = [f[i]['errors'] for i in f.keys()]
    if show_exit_flag:
        a_ef = [f[i]['exit_flag'] for i in f.keys()]
    a = np.array(a_temp)
    
    if plot_error:
        ae = np.array(a_err)
    # print(f'shape of a {a.shape}')



if plot_c:
    with open(f"results_data/Exp{k}_3.json") as file:
        # c_temp = list(json.load(file).values())
        f = json.load(file)
        c_temp = [f[i]['losses'] for i in f.keys()]
        if plot_error:
            c_err = [f[i]['errors'] for i in f.keys()]
        if show_exit_flag:
            c_ef = [f[i]['exit_flag'] for i in f.keys()]
        c = np.array(c_temp)
        if plot_error:
            ce = np.array(c_err)



matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('xtick', labelsize=16)

plt.figure(figsize=(20,5))



mean1 = list(np.nanmean(a, axis=0))
print(len(mean1))
mean1.pop(line)
print(len(mean1))
plt.plot(mean1, '-', label=labels[0], linewidth=3)


mean3 = np.nanmean(c, axis=0)
plt.plot(mean3, '--', label=labels[2], linewidth=3)

plt.vlines(line,min(mean1),max(mean1),linestyles='dotted',colors='gray')

plt.yscale('log')
plt.xlabel('iterations', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.legend(fontsize=20, loc=3)





plt.tight_layout()
plt.savefig('tikzpicture_plots/fig2a_onlyloss.pdf', format="pdf", bbox_inches="tight")
#tikzplotlib.save('tikzpicture_plots/fig2a.tex', axis_height='4cm', axis_width='12cm')
plt.show()
