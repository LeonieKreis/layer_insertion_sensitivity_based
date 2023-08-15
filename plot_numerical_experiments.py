import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import plot_helper
from utils import ema_np

k = 6

std = False
plot_b = True
plot_error = True
plot_area = True
show_exit_flag = True
log_scale = True
smooth = 0.999

if plot_b:
    labels = ['abs max', 'abs min', 'comparison']
if not plot_b:
    labels = ['abs max', 'comparison']

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
    with open(f"results_data/Exp{k}_2.json") as file:
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

if plot_b:
    methods = (a, b, c)
if not plot_b:
    methods = (a, c)

if plot_error:
    if plot_b:
        methods_e = (ae, be, ce)
    if not plot_b:
        methods_e = (ae, ce)

if plot_area:
    if not plot_error:
        fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]})
        m = 1
        n = 3  # egal
    if plot_error:
        fig, axes = plt.subplots(3, gridspec_kw={'height_ratios': [3, 3, 1]})
        m = 2
        n = 3  # egal

if not plot_area:
    if not plot_error:
        fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [10, 1]})
        m = 0
        n = 2  # egal
    if plot_error:
        fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [3, 3]})
        m = 1
        n = 2  # egal

# first subplot plot losses
colors_ = ['b', 'r', 'y']  # , 'g', 'c', 'm', 'k']
for i, aa in enumerate(methods):
    print(aa.shape)
    mean1 = np.nanmean(aa, axis=0)
    if std:
        std1 = np.nanstd(aa, axis=0)

    if smooth is not None:
        mean1 = ema_np(mean1, gamma=smooth)
        if std:
            std1 = ema_np(std1, gamma=smooth)

    if labels is None:
        label = str(i)
    else:
        label = labels[i]
    axes[0].plot(mean1, colors_[i], label=label)
    if std:
        axes[0].plot(mean1 + std1, color=colors_[i], linestyle=':')
        axes[0].plot(np.maximum(mean1 - std1, np.zeros_like(mean1)),
                     color=colors_[i], linestyle=':')

    axes[0].legend()
    #axes[0].set_ylim([0, 2])
    ma = max([a.shape[1] for a in methods])
    axes[0].set_xlim([0, ma])
    axes[0].set_xlabel('iterations')
    axes[0].set_ylabel('average (smoothed) minibatch loss')
    if log_scale:
        axes[0].set_yscale('log')
    axes[0].title.set_text(
        'mean mb-loss averaged over all trainingsruns with dotted std')


# plot errors
if plot_error:
    colors_ = ['b', 'r', 'y']  # , 'g', 'c', 'm', 'k']
    for i, aa in enumerate(methods_e):
        print(aa.shape)
        mean1 = np.nanmean(aa, axis=0)
        if std:
            std1 = np.nanstd(aa, axis=0)
        
        if labels is None:
            label = str(i)
        else:
            label = labels[i]
        axes[1].plot(range(len(mean1)),mean1, colors_[i] + 'o', label=label,
                     markersize=5)  # , linestyle='o')
        if std:
            axes[1].plot(mean1 + std1, colors_[i] + 'o', markersize=2)
            axes[1].plot(np.maximum(mean1 - std1, np.zeros_like(mean1)),
                         colors_[i] + 'o', markersize=2)

        axes[1].legend()
        axes[1].set_ylim([0, 10])
        ma = max([a.shape[1] for a in methods])
        #axes[1].set_xlim([0, ma])
        axes[1].set_xlabel('iterations')
        axes[1].set_ylabel('test error averaged')
        

if plot_area:
    colors = None
    # second and third subplot
    cats_a = plot_helper.translator(a_struct, show=False)
    
    # first a
    x = range(len(cats_a[0]))
    zero = torch.zeros_like(cats_a[0])


    def lines():
        last = zero

        for curve in cats_a:
            next = last + curve
            yield last, next
            last += curve


    if colors is not None:

        for (line1, line2), color in zip(lines(), colors):
            axes[m].fill_betweenx(x, line1, line2, color)

    else:

        for line1, line2 in lines():
            axes[m].fill_between(x, line1, line2)
    # if save_as:
    #     plt.savefig(save_as)
    axes[m].set_ylabel('training runs')
    axes[m].set_xlabel('iterations')
    axes[m].set_xlim([0, ma])
    axes[m].title.set_text(
        'proportion of traningsruns in absmax for each epoch (the colors display the different model) ')

plt.tight_layout()
plt.show()
