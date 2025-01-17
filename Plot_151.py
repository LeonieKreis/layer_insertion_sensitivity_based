import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 151

li_epoch = 450

plot_error=True

log_scale = True


def plot_loss_error_avg(k,li_epoch ,plot_error=True, log_scale=True, no_inits = 10,wrt_time=False, save = None):
    plot_grads = False
    run="0" # for time
    if True:


        with open(f"results_data_spirals/Exp{k}_1.json") as file: #abs max
            f = json.load(file)

            a_temp = [f[i]['losses'] for i in f.keys()]
            for i in range(len(a_temp)):
                if len(a_temp[i])!=18500: print('incorrect lentgh at',i)
                a_temp[i]=a_temp[i]+list(np.zeros(18500-len(a_temp[i])))
            at = f[run]['times'] 
            if plot_error:
                a_err = [f[i]['errors'] for i in f.keys()]
                
            if plot_grads:
                a_grad = f[run]['grad_norms']
            a = np.array(a_temp)
            if plot_error:
                ae = np.array(a_err)
            print(f'shape of a1 {a.shape}')

        



        with open(f"results_data_spirals/Exp{k}_3.json") as file: # baseline
            f = json.load(file)

            a_temp = [f[i]['losses'] for i in f.keys()]
            for i in range(len(a_temp)):
                if len(a_temp[i])!=18500: print('incorrect lentgh at',i)
                a_temp[i]=a_temp[i]+list(np.zeros(18500-len(a_temp[i])))
            at3 = f[run]['times'] 
            if plot_error:
                a_err = [f[i]['errors'] for i in f.keys()]
                
            
            a3 = np.array(a_temp)
            if plot_error:
                ae3 = np.array(a_err)
            print(f'shape of a3 {a3.shape}')


        with open(f"results_data_spirals/Exp{k}_4.json") as file: #fnn2
            f = json.load(file)
            a_temp = [f[i]['losses'] for i in f.keys()]
            for i in range(len(a_temp)):
                if len(a_temp[i])!=18500: print('incorrect lentgh at',i)
                a_temp[i]=a_temp[i]+list(np.zeros(18500-len(a_temp[i])))
            at2 = f[run]['times'] 
            if plot_error:
                a_err = [f[i]['errors'] for i in f.keys()]
            a2 = np.array(a_temp)
            if plot_error:
                ae2 = np.array(a_err)
            print(f'shape of a4 {a2.shape}')

        

        labels = [ 'FNN LI','FNN1','FNN2']
        

        methods = (a,a3,a2)
        times = (at,at3,at2)


        matplotlib.rc('ytick', labelsize=18)
        matplotlib.rc('xtick', labelsize=18)
        fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [5,5]})
        fig.set_size_inches((20,10))

        # first subplot plot losses
        colors_ = ['b','r', 'y', 'g']  # , 'g', 'c', 'm', 'k']
        for i, (aa, ta) in enumerate(zip(methods, times)):
            print(f'i is {i}')
        
            mean1 = np.nanmean(aa, axis=0)
            
            #mean1=aa
            if labels is None:
                label = str(i)
            else:
                label = labels[i]
            if i==0 :
                begin = 0
                end_weg = -1
            elif i==1:
                end_weg = -1
                begin=1
            else:
                end_weg = -1
                begin=0
            #plt.plot(ta[1:end_weg],mean1, colors_[i], label=label)
            print(f'mean1: {mean1}')
            if not wrt_time:
                axes[0].plot(mean1, colors_[i], label=label)
                axes[0].vlines(li_epoch*10, 0, 1, colors='blue', linestyles='dotted')
            else:
                mean1_epoch = mean1[0::10]
                axes[0].plot(ta[begin:end_weg],mean1_epoch, colors_[i], label=label)
                axes[0].vlines(li_epoch, 0, 1, colors='blue', linestyles='dotted')
            axes[0].legend(fontsize=20)
            if wrt_time:
                axes[0].set_xlabel('time', fontsize=20)
            else:
                axes[0].set_xlabel('iterations', fontsize=20)
            axes[0].set_ylabel(' (minibatch) loss', fontsize=20)
            if log_scale:
                axes[0].set_yscale('log')
        


        # plot errors
        if plot_error:
            methods_e = (ae,ae3,ae2)
            colors_ = [ 'b','r','y', 'g']  # , 'g', 'c', 'm', 'k']
            for i, (aa, ta) in enumerate(zip(methods_e, times)):
                print(f'i is {i}')
                mean1 = np.nanmean(aa, axis=0)
                
                #mean1=aa

                if labels is None:
                    label = str(i)
                else:
                    label = labels[i]
                if i==0 or i==2:
                    end_weg = -1
                    begin=0
                elif i==1:
                    begin=1
                    end_weg = -1
                #plt.plot(ta[0:end_weg], mean1, colors_[i] + 'o', label=label,
                #            markersize=5)  # , linestyle='o')
                if not wrt_time:
                    axes[1].plot( mean1, colors_[i] + '-', label=label, linewidth=5)  # , linestyle='o')
                    axes[1].vlines(li_epoch*1, 0, 60, colors='blue', linestyles='dotted')
                else:
                    axes[1].plot(ta[begin:end_weg], mean1[0:-1], colors_[i] + '-', label=label, linewidth=5)
                    axes[1].vlines(li_epoch, 0, 60, colors='blue', linestyles='dotted')
                axes[1].legend(fontsize=20)
                #plt.ylim([0, 100])
                #ma = max([a.shape[1] for a in methods])
                # plt.xlim([0, ma])
                if wrt_time:
                    axes[1].set_xlabel('time', fontsize=20)
                else:
                    axes[1].set_xlabel('epochs', fontsize=20)
                axes[1].set_ylabel('test error', fontsize=20)
            #plt.show()



        fig.tight_layout()
        if save is not None:
            plt.savefig(save, format="pdf", bbox_inches="tight")
        plt.show()

wrt_time = False

if wrt_time:
    li_epoch = 32.1
else:
    li_epoch = 450

save = None #'plots/comp-fixed-architecture-fnns-loss-and-error-mb.pdf'

plot_loss_error_avg(k,li_epoch, plot_error=True, log_scale=True, no_inits = 40,wrt_time=wrt_time, save = save)

#for run in ["0","1","2","3","4","5","6","7","8","9"]:
#    plot_loss_error_single_init(k,li_epoch ,plot_error=True, log_scale=True, run=run, save = True)
    