import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 27

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
if plot_b:
    labels = ['abs max', 'abs min', 'comparison']
if not plot_b:
    labels = ['layer insertion', 'original net']

with open(f"results_data/Exp{k}_1.json") as file:
    # a_temp = list(json.load(file).values())
    f = json.load(file)
    a_temp = [f[i]['losses'] for i in f.keys()]
    a_struct = [f[i]['structures'] for i in f.keys()]
    if plot_error:
        a_err = [f[i]['errors'] for i in f.keys()]
    if plot_grads:
        # a_grad = [f[i]['grad_norms'] for i in f.keys()]
        a_grad = f[run]['grad_norms']
    if show_exit_flag:
        a_ef = [f[i]['exit_flag'] for i in f.keys()]
    a = np.array(a_temp)
    a = a[0:4, :]
    if plot_error:
        ae = np.array(a_err)
    # print(f'shape of a {a.shape}')

if plot_b:
    with open(f"results_data/Exp{k}_4.json") as file:
        # b_temp = list(json.load(file).values())
        f = json.load(file)
        b_temp = [f[i]['losses'] for i in f.keys()]
        b_struct = [f[i]['structures'] for i in f.keys()]
        if plot_error:
            b_err = [f[i]['errors'] for i in f.keys()]
        if plot_grads:
            # b_grad = [f[i]['grad_norms'] for i in f.keys()]
            b_grad = f[run]['grad_norms']
        if show_exit_flag:
            b_ef = [f[i]['exit_flag'] for i in f.keys()]
        b = np.array(b_temp)
        b = b[0:4, :]
        if plot_error:
            be = np.array(b_err)

if plot_c:
    with open(f"results_data/Exp{k}_3.json") as file:
        # c_temp = list(json.load(file).values())
        f = json.load(file)
        c_temp = [f[i]['losses'] for i in f.keys()]
        if plot_error:
            c_err = [f[i]['errors'] for i in f.keys()]
        if plot_grads:
            # c_grad = [f[i]['grad_norms'] for i in f.keys()]
            c_grad = f[run]['grad_norms']
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
    methods = (a, b)
if not plot_b and not plot_c:
    methods = (a)

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
        fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [10, 0]})
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
        if std:
            std1 = ema2_np(std1, gamma=smooth,
                           no_of_steps_back=no_of_steps_included)

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
    # axes[0].set_ylim([0, 2])
    ma = max([a.shape[1] for a in methods])
    axes[0].set_xlim([0, ma])
    # axes[0].set_ylim([0.9, 1.])#ma])
    axes[0].set_xlabel('iterations')
    axes[0].set_ylabel(' (fullbatch) loss')
    if log_scale:
        axes[0].set_yscale('log')
    # axes[0].title.set_text(
    #    'Comparison')


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
        axes[1].plot(range(len(mean1)), mean1, colors_[i] + 'o', label=label,
                     markersize=5)  # , linestyle='o')
        if std:
            axes[1].plot(mean1 + std1, colors_[i] + 'o', markersize=2)
            axes[1].plot(np.maximum(mean1 - std1, np.zeros_like(mean1)),
                         colors_[i] + 'o', markersize=2)

        axes[1].legend()
        axes[1].set_ylim([0, 100])
        ma = max([a.shape[1] for a in methods])
        # axes[1].set_xlim([0, ma])
        axes[1].set_xlabel('iterations')
        axes[1].set_ylabel('test error')

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
    plt.title('Net 1')


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
