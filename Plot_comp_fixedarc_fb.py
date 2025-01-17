import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 6   ## 11 or 6




if k==11:
    no = 2
    net_type = 'resnets'
    timepoint= 21.5 #23.8
    minimum_l=0.5
    maximum_l=0.8
    minimum_e = 30
    maximum_e = 60
if k==6:
    no = 1
    net_type = 'fnns'
    timepoint=23.8
    minimum_l=0.25
    maximum_l=0.65
    minimum_e = 15
    maximum_e = 50

li_epoch = 450

save = None#f'plots/comp-fixed-architecture-{net_type}-loss-and-error.pdf'
savesingleloss = None#f'plots/comp-fixed-architecture-{net_type}-loss-for-grads-fb.pdf'
savegrads = None#f'plots/comp-fixed-architecture-{net_type}-grads-fb.pdf'


run = "0"

plot_error=True
plot_grads = True
log_scale = True
avg = None

with open(f"results_data_spirals/Exp{k}_1.json") as file:
    f = json.load(file)
    a_temp = [f[i]['losses'] for i in f.keys()]
    a_loss_grad = f[run]['losses']
    at = f[run]['times'] 
    if plot_error:
        a_err = [f[i]['errors'] for i in f.keys()]
    if plot_grads:
        a_grad = f[run]['grad_norms']
    a = np.array(a_temp)
    if plot_error:
        ae = np.array(a_err)
    print(f'shape of a {a.shape}')


with open(f"results_data_spirals/Exp{k}_3.json") as file:
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
    print(f'shape of c {c.shape}')
    print(len(ct))

with open(f"results_data_spirals/Exp{k}_4.json") as file:
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
    print(f'shape of d {d.shape}')
    print(len(dt))

if k==11:
    labels = ['ResNet LI','ResNet1','ResNet2']
if k==6:
    labels = ['FNN LI','FNN1','FNN2']

methods = (a,c,d)
times = (at,ct,dt)

matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('xtick', labelsize=16)
fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [5,5]})
fig.set_size_inches((20,10))

# first subplot plot losses
colors_ = ['b', 'r', 'y','g']  # , 'g', 'c', 'm', 'k']
for i, (aa, ta) in enumerate(zip(methods, times)):
    print(aa.shape)
    mean1 = np.nanmean(aa, axis=0)
    if labels is None:
        label = str(i)
    else:
        label = labels[i]
    if i==0:
        end_weg = -1
    else:
        end_weg = None
    
    #axes[0].plot(ta[1:end_weg],mean1, colors_[i], label=label)
    axes[0].plot(mean1, colors_[i], label=label)
    axes[0].legend(fontsize=20)
    #plt.ylim([0, 2])
    ma = max([a.shape[1] for a in methods])
    axes[0].vlines(li_epoch,minimum_l,maximum_l,linestyles='dotted',colors='blue')
    #plt.xlim([0, ma])
    axes[0].set_xlabel('iterations', fontsize=20)
    axes[0].set_ylabel(' loss', fontsize=20)
    if log_scale:
        axes[0].set_yscale('log')




# plot errors
if plot_error:
    methods_e = (ae,ce,de)
    
    colors_ = ['b', 'r', 'y','g']  # , 'g', 'c', 'm', 'k']
    for i, (aa, ta) in enumerate(zip(methods_e, times)):
        print(f'aa.shape {aa.shape}')
        mean1 = np.nanmean(aa, axis=0)

        if labels is None:
            label = str(i)
        else:
            label = labels[i]
        if i==0:
            end_weg = -1
        else:
            end_weg = None
    
        # axes[1].plot(ta[0:end_weg],mean1, colors_[i] , label=label,
        #              linewidth=2)  # , linestyle='o')
        axes[1].plot(mean1, colors_[i] , label=label,
                     linewidth=2)  # , linestyle='o')
        
        axes[1].vlines(li_epoch,minimum_e,maximum_e,linestyles='dotted',colors='blue')
    
        axes[1].legend(fontsize=20)
        #plt.ylim([0, 60])
        ma = max([a.shape[1] for a in methods])
        
        # plt.xlim([0, ma])
        axes[1].set_xlabel('epochs', fontsize=20)
        axes[1].set_ylabel('test error (%)', fontsize=20)
    #plt.show()

plt.tight_layout()
if save is not None:
    plt.savefig(save, format="pdf", bbox_inches="tight")
plt.show()

if plot_grads and k==6:
    labels0 = ['W1','b1', 'W2','b2']
    labels1 = ['W1','b1','Wnew','bnew','W2','b2']
    colors0=['b','b','g','g']
    colors1=['b','b','r','r','g','g']
    # optional:remove biases from grad structure?
    plt.figure(figsize=(20,5))
    if avg is not None:
        a_loss_grad = ema_np(a_loss_grad, avg)
    plt.plot(a_loss_grad, 'b',label='FNN LI')
    plt.vlines(li_epoch, 0.2, 1, colors='blue', linestyles='dashed')
    plt.legend(fontsize=20)
    plt.yscale('log')
    plt.xlabel('iterations', fontsize=20)
    plt.ylabel('loss', fontsize=20)

    plt.tight_layout()
    if savesingleloss is not None:
        plt.savefig(savesingleloss, format="pdf", bbox_inches="tight")
    plt.show()
    
    plt.figure(figsize=(20,5))
    for i, pg in enumerate(a_grad[0]):
        if avg is not None:
            pg = ema2_np(pg, avg)
        if i%2==0:
            print(i)
            plt.plot(pg,colors0[i], label=labels0[i])
        else:
            continue
    no_its0 = len(a_grad[0][0])
    no_its1 = len(a_grad[1][0])
    for i,pg in enumerate(a_grad[1]):
        if avg is not None:
            pg = ema2_np(pg, avg)
        if i%2==0:
            if i==2:
                print(i)
                plt.plot(list(range(no_its0, no_its0 + no_its1)),pg,colors1[i], label='Wnew')
            else:
                print(i)
                plt.plot(list(range(no_its0, no_its0 + no_its1)),pg,colors1[i])
        else:
            continue
    plt.vlines(li_epoch, 10e-6, 10e-3, colors='blue', linestyles='dashed')
    plt.legend( fontsize=20)
    plt.yscale('log')
    plt.xlabel('iterations', fontsize=20)
    plt.ylabel('weight grad norms', fontsize=20)


plt.tight_layout()
if savegrads is not None:
    plt.savefig(savegrads, format="pdf", bbox_inches="tight")
plt.show()

# comp-fixed-architecture-resnets-loss-and-error.pdf