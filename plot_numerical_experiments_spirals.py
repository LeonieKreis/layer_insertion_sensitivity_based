import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 1

run = "0"

std = False
plot_grads = True
plot_b = True
plot_c = True
plot_error = True
plot_area = False
show_exit_flag = False
log_scale = True
smooth = 0.
no_of_steps_included = 1000


with open(f"results_data_spirals/Exp{k}_1.json") as file:
    f = json.load(file)
    a_temp = [f[i]['losses'] for i in f.keys()]
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
    b_temp = [f[i]['losses'] for i in f.keys()]
    if plot_error:
        b_err = [f[i]['errors'] for i in f.keys()]
    if plot_grads:
        b_grad = f[run]['grad_norms']
    b = np.array(b_temp)
    if plot_error:
        be = np.array(b_err)
    # print(f'shape of b {b.shape}')

with open(f"results_data_spirals/Exp{k}_3.json") as file:
    f = json.load(file)
    c_temp = [f[i]['losses'] for i in f.keys()]
    if plot_error:
        c_err = [f[i]['errors'] for i in f.keys()]
    if plot_grads:
        c_grad = f[run]['grad_norms']
    c = np.array(c_temp)
    if plot_error:
        ce = np.array(c_err)
    # print(f'shape of c {c.shape}')

with open(f"results_data_spirals/Exp{k}_4.json") as file:
    f = json.load(file)
    d_temp = [f[i]['losses'] for i in f.keys()]
    if plot_error:
        d_err = [f[i]['errors'] for i in f.keys()]
    if plot_grads:
        d_grad = f[run]['grad_norms']
    d = np.array(d_temp)
    if plot_error:
        de = np.array(d_err)
    # print(f'shape of d {d.shape}')


methods = (a,b,c,d)
labels = ['LI', 'LI2','N1','N2']
plt.figure(figsize=(20,5))

# first subplot plot losses
colors_ = ['b', 'r', 'y','g']  # , 'g', 'c', 'm', 'k']
for i, aa in enumerate(methods):
    print(aa.shape)
    mean1 = np.nanmean(aa, axis=0)
    if std:
        std1 = np.nanstd(aa, axis=0)

    if smooth is not None:
        if std:
            std1 = ema2_np(std1, gamma=smooth,
                           no_of_steps_back=no_of_steps_included)

    if labels is None:
        label = str(i)
    else:
        label = labels[i]
    plt.plot(mean1, colors_[i], label=label)
    if std:
        plt.plot(mean1 + std1, color=colors_[i], linestyle=':')
        plt.plot(np.maximum(mean1 - std1, np.zeros_like(mean1)),
                     color=colors_[i], linestyle=':')

    plt.legend()
    #plt.ylim([0, 2])
    ma = max([a.shape[1] for a in methods])
    plt.xlim([0, ma])
    plt.xlabel('iterations')
    plt.ylabel(' (fullbatch) loss')
    if log_scale:
        plt.yscale('log')



# plot errors
if plot_error:
    methods_e = (ae,be,ce,de)
    plt.figure(figsize=(20,5))
    colors_ = ['b', 'r', 'y','g']  # , 'g', 'c', 'm', 'k']
    for i, aa in enumerate(methods_e):
        #print(aa.shape)
        mean1 = np.nanmean(aa, axis=0)
        if std:
            std1 = np.nanstd(aa, axis=0)

        if labels is None:
            label = str(i)
        else:
            label = labels[i]
        plt.plot(range(len(mean1)), mean1, colors_[i] + 'o', label=label,
                     markersize=5)  # , linestyle='o')
        if std:
            plt.plot(mean1 + std1, colors_[i] + 'o', markersize=2)
            plt.plot(np.maximum(mean1 - std1, np.zeros_like(mean1)),
                         colors_[i] + 'o', markersize=2)

        plt.legend()
        plt.ylim([0, 100])
        ma = max([a.shape[1] for a in methods])
        # plt.xlim([0, ma])
        plt.xlabel('iterations')
        plt.ylabel('test error')
    #plt.show()

if plot_grads:  # works only for layer insertion once
    # only for index run!!
    grad_norms1 = a_grad
    plt.figure(figsize=(20, 5))
    l1 = len(grad_norms1[0])
    l2 = len(grad_norms1[1])
    len_t1 = len(grad_norms1[0][0])
    len_t2 = len(grad_norms1[1][0])
    for i in range(l1):
        plt.plot(grad_norms1[0][i], label=f'0_{i}')
    for j in range(l2):
        # print(list(range(len_t1,len_t2+len_t1)))
        plt.plot(list(range(len_t1, len_t2+len_t1)),
                 grad_norms1[1][j], label=f'1_{j}')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('layerwise gradient norms scaled by lr')
    plt.title('LI')


if plot_grads:
    # only for index run!!
    grad_norms2 = c_grad
    plt.figure(figsize=(20, 5))
    l1 = len(grad_norms2)
    print(l1)
    # print(len(grad_norms2[0][0]))
    # l2 = len(grad_norms2[1])
    # print(l2)
    for i in range(l1):
        plt.plot(grad_norms2[i], label=f'0_{i}')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('layerwise gradient norms scaled by lr')
    plt.title('N1')


plt.tight_layout()
plt.show()
