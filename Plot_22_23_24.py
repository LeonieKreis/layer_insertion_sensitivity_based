import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 24 # 22 23 24 possible



plot_error=True
plot_grads= True
log_scale = True


def plot_loss_error_single_init(k,li_epoch ,plot_error=True,plot_grads=False, log_scale=True, run="0", save = False):
    avg = 0.99 # 0.9
    if True:
        with open(f"results_data_spirals/Exp{k}_1.json") as file: #abs max
            f = json.load(file)

            a_temp = f[run]['losses'] 
            at = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = f[run]['errors']
            if plot_grads:
                a_grad = f[run]['grad_norms']
            a = np.array(a_temp)
            if plot_error:
                ae = np.array(a_err)
            print(f'shape of a {a.shape}')
            a_sens = f[run]['sens']


        with open(f"results_data_spirals/Exp{k}_3.json") as file: # baseline
            f = json.load(file)

            a_temp = f[run]['losses'] 
            at3 = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = f[run]['errors']
            
            a3 = np.array(a_temp)
            if plot_error:
                ae3 = np.array(a_err)
            print(f'shape of a {a3.shape}')


        

        labels = ['LI', 'baseline']

        methods = (a,a3)
        times = (at,at3)


        plt.figure(figsize=(20,5))

        # first subplot plot losses
        colors_ = ['b', 'r', 'y','g']  # , 'g', 'c', 'm', 'k']
        for i, (aa, ta) in enumerate(zip(methods, times)):
            print(aa.shape)
            #mean1 = np.nanmean(aa, axis=0)
            mean1=aa
            if labels is None:
                label = str(i)
            else:
                label = labels[i]
            if i==0:
                end_weg = -1
            else:
                end_weg = None
            #plt.plot(ta[1:end_weg],mean1, colors_[i], label=label)
            plt.plot(mean1, colors_[i], label=label)
            plt.vlines(li_epoch*10, 0, 1, colors='gray', linestyles='dashed')
            plt.legend()
            #plt.ylim([0, 2])
            #ma = max([a.shape[1] for a in methods])
            #plt.xlim([0, ma])
            plt.xlabel('its')
            plt.ylabel(' (minibatch) loss')
            plt.title(f'with sensitivities {a_sens}')
            if log_scale:
                plt.yscale('log')
        if save==True:
            plt.savefig(f'figs/mbloss_{k}_{run}.pdf', format="pdf", bbox_inches="tight")



        # plot errors
        if plot_error:
            methods_e = (ae,ae3)
            plt.figure(figsize=(20,5))
            colors_ = ['b', 'r', 'y','g']  # , 'g', 'c', 'm', 'k']
            for i, (aa, ta) in enumerate(zip(methods_e, times)):
                #print(aa.shape)
                #mean1 = np.nanmean(aa, axis=0)
                mean1=aa

                if labels is None:
                    label = str(i)
                else:
                    label = labels[i]
                if i==0:
                    end_weg = -1
                else:
                    end_weg = None
                #plt.plot(ta[0:end_weg], mean1, colors_[i] + 'o', label=label,
                #            markersize=5)  # , linestyle='o')
                plt.plot( mean1, colors_[i] + 'o', label=label,
                            markersize=5)  # , linestyle='o')
                plt.vlines(li_epoch, 0, 60, colors='gray', linestyles='dashed')
                plt.legend()
                #plt.ylim([0, 100])
                #ma = max([a.shape[1] for a in methods])
                # plt.xlim([0, ma])
                plt.xlabel('epochs')
                plt.ylabel('test error')
            #plt.show()

        if plot_grads:
            labels1 = ['W1','b1','W2','b2','W3','b3','W4']
            # optional:remove biases from grad structure?
            
            plt.figure(figsize=(20,5))
            for i, pg in enumerate(a_grad[1]):
                if avg is not None:
                    pg = ema2_np(pg, avg)
                plt.plot(pg, label=labels1[i])
            
            plt.legend()
            plt.yscale('log')
            plt.xlabel('iterations')
            plt.ylabel('weight grad norms')
            



        plt.tight_layout()
        if save==True and plot_error:
            plt.savefig(f'figs/testerror_{k}_{run}.pdf', format="pdf", bbox_inches="tight")
        #plt.savefig('tikzpicture_plots/fig_27.pdf', format="pdf", bbox_inches="tight")
        plt.show()

ks = [22,23,24]
li_epochs = [0,0,0]

plot_loss_error_single_init(k, 0, plot_error,plot_grads, log_scale, run="0", save = False)