import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib
import torch
import plot_helper
from utils import ema_np, ema2_np

k =  9# 9 for a or 14 for b
line = 70 # 70 or 20

std = False
plot_b = True # abs min

plot_error = True
plot_area = False
show_exit_flag = False
log_scale = True
smooth = 0. # not needed for GD
no_of_steps_included = 1000
if plot_b:
    labels = ['Layer insertion', 'Alternative']

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

if plot_b:
    with open(f"results_data/Exp{k}_2.json") as file: #Exp8_3
        # b_temp = list(json.load(file).values())
        f = json.load(file)
        b_temp = [f[i]['losses'] for i in f.keys()]
        b_struct = [f[i]['structures'] for i in f.keys()]
        if plot_error:
            b_err = [f[i]['errors'] for i in f.keys()]
        if show_exit_flag:
            b_ef = [f[i]['exit_flag'] for i in f.keys()]
        b = np.array(b_temp)
        
        if plot_error:
            be = np.array(b_err)





matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('xtick', labelsize=16)
plt.figure(figsize=(20,5))



mean1 = list(np.nanmean(a, axis=0))
print(len(mean1))
mean1.pop(line)
print(len(mean1))
#plt.plot(mean1, '-', label=labels[0])

mean2 = list(np.nanmean(b, axis=0))
print(len(mean2))
mean2.pop(line)
print(len(mean2))
#plt.plot(mean2, '--', label=labels[1])

diff1 = [mean2[i]-mean1[i] for i in range(len(mean1))]
#plt.plot(list(range(len(mean1))),diff1, label = 'FNN1-LI', linewidth=3)
#print(diff1)



#plt.vlines(line,min(mean1),max(mean1),linestyles='dotted',colors='gray')

#plt.yscale('log')
#plt.xlabel('iterations', fontsize=20)
#plt.ylabel('loss', fontsize=20)
#plt.ylim(bottom=-0.07)
#plt.legend(fontsize=20)





#plt.tight_layout()
#plt.savefig('tikzpicture_plots/fig5a_diff.pdf', format="pdf", bbox_inches="tight")
#tikzplotlib.save('tikzpicture_plots/fig5b.tex', axis_height='4cm', axis_width='12cm')
#plt.show()
