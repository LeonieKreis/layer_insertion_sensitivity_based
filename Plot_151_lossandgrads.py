import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 151



plot_error=True

log_scale = True
 


def plot_loss_error_run(run,k,li_epoch ,plot_grads=True, log_scale=True,avg=None, save = None):

    if True:


        with open(f"results_data_spirals/Exp{k}_1.json") as file: #abs max
            f = json.load(file)

            a_temp = f[run]['losses'] 
            if len(a_temp)!=18500: 
                print('incorrect lentgh at',len(a_temp))
                #a_temp=a_temp+list(np.zeros(18500-len(a_temp)))
            at = f[run]['times']   
            if plot_grads:
                a_grad = f[run]['grad_norms']
            a = np.array(a_temp)

        



      

        labels = [ 'FNN LI','FNN LI']
        colors = ['b','b']  # , 'g', 'c', 'm', 'k']

        methods = (a,a)
        times = (at,at)


        matplotlib.rc('ytick', labelsize=18)
        matplotlib.rc('xtick', labelsize=18)
        fig, axes = plt.subplots(1, gridspec_kw={'height_ratios': [5]})
        fig.set_size_inches((20,5))

        # first subplot plot losses
        colors_ = [ 'r','b', 'y','g']  # , 'g', 'c', 'm', 'k']
        # for i, (aa, ta) in enumerate(zip(methods, times)):
        #     print(aa.shape)
            
        #     #mean1 = np.nanmean(aa, axis=0)
        #     #print(mean1.shape)
            
        #     #print(aa.shape)
        #     mean1=aa
        #     if labels is None:
        #         label = str(i)
        #     else:
        #         label = labels[i]
        #     if i==0:
        #         end_weg = -1
        #     else:
        #         end_weg = None
        #     #plt.plot(ta[1:end_weg],mean1, colors_[i], label=label)
        #     print(f'mean1: {mean1}')
        #     if avg is not None:
        #         mean1 = ema2_np(mean1, avg)
        #     if i==0:
        #         axes[0].plot(mean1, colors_[i])
        #     else:
        #         axes[0].plot(mean1, colors_[i], label=label)
        #     axes[0].vlines(li_epoch*10, 0.2, 1, colors='gray', linestyles='dashed')
        #     axes[0].legend(fontsize=20)
        #     #plt.ylim([0, 2])
        #     #ma = max([a.shape[1] for a in methods])
        #     #plt.xlim([0, ma])
        #     axes[0].set_xlabel('iterations', fontsize=20)
        #     axes[0].set_ylabel(' (minibatch) loss', fontsize=20)
        #     if log_scale:
        #         axes[0].set_yscale('log')
        
        if plot_grads:
            plot_bias = False
            li_epoch = 450
            labels1=['W1','b1','W2']
            labels2 = ['W1','b1','W2','b2','W3']
            colors1 = ['r','r','g','g']
            colors2 = ['r','r','b','b','g','g']
            for i,g in enumerate(a_grad[0]):
                if avg is not None:
                    g = ema2_np(g, avg)
                if not plot_bias and i==1:
                    continue
                else:
                    axes.plot(g,colors1[i], label=labels1[i])
            axes.vlines(li_epoch*10, 10e-5, 10e-3, colors='blue', linestyles='dotted')
            its2 = list(range(li_epoch*10, li_epoch*10+len(a_grad[1][0])))
            for j, gg in enumerate(a_grad[1]):
                if avg is not None:
                    gg = ema2_np(gg, avg)
                if not plot_bias and j%2==1:
                    continue
                else:
                    if j==2:
                        axes.plot(its2,gg, colors2[j],label='Wnew')
                    else:
                        axes.plot(its2,gg, colors2[j])

            axes.set_yscale('log')
            axes.set_xlabel('iterations', fontsize=20)
            axes.set_ylabel('weight gradient norms', fontsize=20)
            axes.legend(fontsize=20)


        fig.tight_layout()
        if save is not None:
            plt.savefig(save, format="pdf", bbox_inches="tight")
        #plt.savefig('tikzpicture_plots/fig_27.pdf', format="pdf", bbox_inches="tight")
        plt.show()

avg = 0.999#None
run ="16"

li_epoch = 450
save = None #'plots/loss-and-layerwise-grads-fnn-mb.pdf'

plot_loss_error_run(run,k,li_epoch, plot_grads=True, log_scale=True,avg=avg, save = save)

