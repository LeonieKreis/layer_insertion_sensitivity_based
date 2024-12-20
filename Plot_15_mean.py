import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 15100#15 #151 for new sens comp

run = "9"
i=run

plot_error=True
plot_grads = False
log_scale = True

save = False

def plot_loss_error_mean(plot_error, plot_grads, log_scale, run,k, save = False):
    
    if True:
        with open(f"results_data_spirals/Exp{k}_1.json") as file: #abs max
            f = json.load(file)
            a_temp = [f[i]['losses'] for i in f.keys()] 
            for i in range(len(a_temp)):
                if len(a_temp[i])!=18500: a_temp[i]=a_temp[i]+list(np.zeros(18500-len(a_temp[i])))
            at = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = [f[i]['errors'] for i in f.keys()]
            if plot_grads:
                a_grad = f[run]['grad_norms']
            a = np.array(a_temp)
            if plot_error:
                ae = np.array(a_err)
            print(f'shape of a {a.shape}')

        with open(f"results_data_spirals/Exp{k}_2.json") as file: # abs min
            f = json.load(file)
            a_temp = [f[i]['losses'] for i in f.keys()]
            for i in range(len(a_temp)):
                if len(a_temp[i])!=18500: a_temp[i]=a_temp[i]+list(np.zeros(18500-len(a_temp[i])))
            at2 = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = [f[i]['errors'] for i in f.keys()]
            if plot_grads:
                a_grad = f[run]['grad_norms']
            a2 = np.array(a_temp)
            if plot_error:
                ae2 = np.array(a_err)
            print(f'shape of a {a2.shape}')

        with open(f"results_data_spirals/Exp{k}_3.json") as file: # baseline
            f = json.load(file)
            a_temp = [f[i]['losses'] for i in f.keys()]
            at3 = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = [f[i]['errors'] for i in f.keys()]
            if plot_grads:
                a_grad = f[run]['grad_norms']
            a3 = np.array(a_temp)
            if plot_error:
                ae3 = np.array(a_err)
            print(f'shape of a {a3.shape}')


        with open(f"results_data_spirals/Exp{k}_4.json") as file: #big cnn
            f = json.load(file)
            a_temp = [f[i]['losses'] for i in f.keys()]
            a_temp[1]=a_temp[1]+list(np.zeros(18500-len(a_temp[1])))
            at4 = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = [f[i]['errors'] for i in f.keys()]
                
                for i in range(10):
                    if len(a_err[i])!=18500: print('incorrect lentgh at',i)
            if plot_grads:
                a_grad = f[run]['grad_norms']
            a4 = np.array(a_temp)
            if plot_error:
                ae4 = np.array(a_err)
            print(f'shape of a {a4.shape}')
    

        labels = ['LI', 'FNN1','FNN2']

        methods = (a,a3,a4)
        times = (at,at3,at4)

        matplotlib.rc('ytick', labelsize=12)
        matplotlib.rc('xtick', labelsize=12)
        fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [5,5]})
        fig.set_size_inches((20,10))
        
        # first subplot plot losses
        colors_ = ['b', 'r', 'y','g']  # , 'g', 'c', 'm', 'k']
        for i, (aa, ta) in enumerate(zip(methods, times)):
            print(aa.shape)
            mean1 = np.nanmean(aa, axis=0)
            #mean1=aa
            if labels is None:
                label = str(i)
            else:
                label = labels[i]
            if i==0:
                end_weg = -1
            else:
                end_weg = None
            #plt.plot(ta[1:end_weg],mean1, colors_[i], label=label)
            axes[0].plot(mean1, colors_[i], label=label)
            axes[0].vlines(450*10, 0, 1, colors='gray', linestyles='dashed')
            axes[0].legend()
            #plt.ylim([0, 2])
            #ma = max([a.shape[1] for a in methods])
            #plt.xlim([0, ma])
            axes[0].set_xlabel('its')
            axes[0].set_ylabel(' (minibatch) loss')
            if log_scale:
                axes[0].set_yscale('log')
        


        # plot errors
        if plot_error:
            methods_e = (ae,ae3,ae4)
            colors_ = ['b', 'r', 'y','g']  # , 'g', 'c', 'm', 'k']
            for i, (aa, ta) in enumerate(zip(methods_e, times)):
                #print(aa.shape)
                mean1 = np.nanmean(aa, axis=0)
                #mean1=aa

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
                axes[1].plot( mean1, colors_[i] + 'o', label=label,
                            markersize=5)  # , linestyle='o')
                axes[1].vlines(450, 0, 100, colors='gray', linestyles='dashed')
                axes[1].legend()
                axes[1].set_ylim([0, 100])
                #ma = max([a.shape[1] for a in methods])
                # plt.xlim([0, ma])
                axes[1].set_xlabel('epochs')
                axes[1].set_ylabel('test error')
            #plt.show()



        fig.tight_layout()
        if save==True and plot_error:
            fig.savefig(f'figs/testerror_{k}_{run}.pdf', format="pdf", bbox_inches="tight")
        #plt.savefig('tikzpicture_plots/fig_27.pdf', format="pdf", bbox_inches="tight")
        plt.show()
        
plot_loss_error_mean(plot_error, plot_grads, log_scale, run,k, save)