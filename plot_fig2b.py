import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 15 #11 or 12 or 14, we use k=14
line= 20 # 70

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
    labels = ['Layer insertion', 'ResNet2', 'ResNet1'] # a
    #labels = ['Layer insertion', 'ResNet1', 'ResNet2'] #b

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
    with open(f"results_data/Exp{k}_4.json") as file: #Exp8_3 or k_4
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

if plot_b and plot_c:
    methods = (a, b, c)
if not plot_b and plot_c:
    methods = (a, c)
if plot_b and not plot_c:
    methods=(a,b)
if not plot_b and not plot_c:
    methods=(a)

if plot_error:
    if plot_b:
        methods_e = (ae, be, ce)
    if not plot_b:
        methods_e = (ae, ce)



matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('xtick', labelsize=16)
fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [3, 3]})
fig.set_size_inches(20,10)



mean1 = list(np.nanmean(a, axis=0))
print(len(mean1))
mean1.pop(line)
print(len(mean1))
axes[0].plot(mean1, '-', label=labels[0])

mean2 = np.nanmean(b, axis=0)
axes[0].plot(mean2, '--', label=labels[1])

mean3 = np.nanmean(c, axis=0)
axes[0].plot(mean3, ':', label=labels[2])

axes[0].vlines(line,min(mean1),max(mean1),linestyles='dotted',colors='gray')

axes[0].set_yscale('log')
axes[0].set_xlabel('iterations', fontsize=20)
axes[0].set_ylabel('loss', fontsize=20)
axes[0].legend(fontsize=20, loc=1)



if plot_error:
    
    mean1 = np.nanmean(ae, axis=0)
    axes[1].plot(range(len(mean1)),mean1, 'o', label=labels[0],
                        markersize=3) 
    
    mean2 = np.nanmean(be, axis=0)
    axes[1].plot(range(len(mean2)),mean2, 'x', label=labels[1],
                        markersize=3) 
    
    mean3 = np.nanmean(ce, axis=0)
    axes[1].plot(range(len(mean3)),mean3, 's', label=labels[2],
                        markersize=3) 
    
    axes[1].vlines(line,0.9,100,linestyles='dotted',colors='gray')

    axes[1].set_xlabel('iterations', fontsize=20)
    axes[1].set_ylabel('test error', fontsize=20)
    axes[1].legend(fontsize=20, loc=1)





plt.tight_layout()
plt.savefig('tikzpicture_plots/fig2b.pdf', format="pdf", bbox_inches="tight")
#tikzplotlib.save('tikzpicture_plots/fig2a.tex', axis_height='4cm', axis_width='12cm')
plt.show()
