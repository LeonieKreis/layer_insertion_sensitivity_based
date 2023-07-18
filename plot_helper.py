from matplotlib import pyplot as plt
import numpy as np
import torch
from utils import ema_np


def areaplot(*curves, colors=None, save_as=None, show=True):

    x = range(len(curves[0]))

    zero = torch.zeros_like(curves[0])
    # top = len(curves) * torch.ones_like(curves[0])

    def lines():
        last = zero

        for curve in curves:
            next = last + curve
            yield last, next
            last += curve

        # yield curves[-1], top

    if colors is not None:

        for (line1, line2), color in zip(lines(), colors):
            plt.fill_betweenx(x, line1, line2, color)

    else:

        for line1, line2 in lines():
            plt.fill_between(x, line1, line2)
    # if save_as:
    #     plt.savefig(save_as)
    plt.ylabel('training runs')
    plt.xlabel('epochs')
    plt.title(
        'proportion of traningsruns in model for each epoch (the colors display the different model) ')
    if show:
        plt.show()


def translator(data, show=True):

    max_n_cats = max(len(d) for d in data)
    max_n_epochs = max(sum(d) for d in data)

    cats = torch.zeros(max_n_cats, max_n_epochs, dtype=torch.int32)

    for d in data:
        curr_ind = 0
        for k, it in enumerate(d):
            cats[k, curr_ind:curr_ind + it].add_(1)
            curr_ind += it

    # print(len(cats))
    # print(cats)
    if show is False:
        return cats
    areaplot(*cats, show=show)


def plot_statistics(*methods, _type='loss', show=True, loss=True, smooth=None):
    colors = ['b', 'r', 'y', 'g', 'c', 'm', 'k']
    for i, a in enumerate(methods):
        mean1 = np.nanmean(a, axis=0)
        std1 = np.nanstd(a, axis=0)

        if smooth is not None:
            mean1 = ema_np(mean1, gamma=smooth)
            std1 = ema_np(std1, gamma=smooth)

        plt.plot(mean1, label=str(i), color=colors[i])
        plt.plot(mean1+std1, colors[i]+'--')
        plt.plot(np.maximum(mean1-std1, np.zeros_like(mean1)), colors[i]+'--')

    plt.legend()
    if loss:
        plt.ylim((0, 1))
        plt.ylabel('loss')
    # if _type == 'error':
    #     plt.ylim((0, 60))
    #     plt.ylabel('error')
    m = max([a.shape[1] for a in methods])
    plt.xlim((0, m))
    if show:
        plt.show()


def plot_all(*methods):
    for a in methods:  # iterate over all methods in the comparsion
        # iterate over all trainingsruns in current method a
        for k in range(a.shape[0]):
            plt.plot(a[k, :])
        plt.ylim((0, 1))
        plt.show()

def getx_coords_of_error(end_list, interval_testerror, train='ali'):
    l = []
    l_start = 0
    for e in end_list:
        for i in range(e):
            if i % interval_testerror == 0:
                l.append(l_start+i)
        l_start += e
    if train == 'ali':
        l.append(l_start)
        return l
    if train == 'classical':
        return l
