import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 9  # 9 for feedforward nets or 14 for resnets

if k==9:
    no=1
    net_type= 'fnns'
    minimum_l = 0.35
    maximum_l = 0.7
if k==14:
    no=2
    net_type = 'resnets'
    minimum_l = 0.5
    maximum_l = 0.8
#if k==13:
#    no=3
#    net_type = 'resnets_old'

run = "0"

plot_error=False
plot_grads = True
log_scale = True

with open(f"results_data_spirals/Exp{k}_1.json") as file:
    f = json.load(file)
    a_temp = [f[i]['losses'] for i in f.keys()]
    at = f[run]['times'] 
    if plot_error:
        a_err = [f[i]['errors'] for i in f.keys()]
    if plot_grads:
        a_grad = f[run]['grad_norms']
    a = np.array(a_temp)
    if plot_error:
        ae = np.array(a_err)
    # print(f'shape of a {a.shape}')


with open(f"results_data_spirals/Exp{k}_2.json") as file:
    f = json.load(file)
    c_temp = [f[i]['losses'] for i in f.keys()]
    ct = f[run]['times'] 
    if plot_error:
        c_err = [f[i]['errors'] for i in f.keys()]
    if plot_grads:
        c_grad = f[run]['grad_norms']
    c = np.array(c_temp)
    if plot_error:
        ce = np.array(c_err)
    # print(f'shape of c {c.shape}')

with open(f"results_data_spirals/Exp{k}_3.json") as file:
    f = json.load(file)
    d_temp = [f[i]['losses'] for i in f.keys()]
    dt = f[run]['times'] 
    if plot_error:
        d_err = [f[i]['errors'] for i in f.keys()]
    if plot_grads:
        d_grad = f[run]['grad_norms']
    d = np.array(d_temp)
    if plot_error:
        de = np.array(d_err)
    # print(f'shape of d {d.shape}')

labels = ['LI','LI2']#,'N1']

methods = (a,c)#,d)
times = (at,ct)#,dt)

matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('xtick', labelsize=16)
plt.figure(figsize=(20,5))

# first subplot plot losses
colors_ = ['b', 'r', 'y','g']  # , 'g', 'c', 'm', 'k']
for i, (aa, ta) in enumerate(zip(methods, times)):
    print(aa.shape)
    mean1 = np.nanmean(aa, axis=0)
    if labels is None:
        label = str(i)
    else:
        label = labels[i]
    if i==0 or i==1:
        end_weg = -1
    else:
        end_weg = None
    plt.plot(mean1[450:452+3], colors_[i], label=label)
    #plt.vlines(450,minimum_l,maximum_l,linestyles='dotted',colors='gray')
    plt.legend(fontsize=15)
    #plt.ylim([0, 2])
    ma = max([a.shape[1] for a in methods])
    #plt.xlim([0, ma])
    plt.xlabel('iterations', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    if log_scale:
        plt.yscale('log')

plt.tight_layout()
#plt.savefig(f'../../Papers/plots/validity-LI-{net_type}-loss-closeup.pdf', format="pdf", bbox_inches="tight")


plt.show()

