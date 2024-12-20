import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 25 

li_epoch = 0

plot_error=True

log_scale = True


def plot_loss_error_single_init(k,li_epoch ,plot_error=True, log_scale=True, run="0", save = False):
    plot_grads = False
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

        with open(f"results_data_spirals/Exp{k}_2.json") as file: #abs min

            a_temp = f[run]['losses'] 
            at2 = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = f[run]['errors']
            if plot_grads:
                a_grad = f[run]['grad_norms']
            a2 = np.array(a_temp)
            if plot_error:
                ae2 = np.array(a_err)
            print(f'shape of a {a.shape}')
            a_sens2 = f[run]['sens']



        with open(f"results_data_spirals/Exp{k}_3.json") as file: # baseline
            f = json.load(file)

            a_temp = f[run]['losses'] 
            at3 = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = f[run]['errors']
            if plot_grads:
                a_grad = f[run]['grad_norms']
            a3 = np.array(a_temp)
            if plot_error:
                ae3 = np.array(a_err)
            print(f'shape of a {a3.shape}')


        

        labels = [ 'LImin','LI','baseline']
        colors = ['b', 'r', 'y','g']  # , 'g', 'c', 'm', 'k']

        methods = (a2,a,a3)
        times = (at2,at,at3)


        plt.figure(figsize=(20,5))

        # first subplot plot losses
        colors_ = [ 'r','b', 'y','g']  # , 'g', 'c', 'm', 'k']
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
            plt.title(f'sensitivities: {a_sens}')
            plt.xlabel('its')
            plt.ylabel(' (minibatch) loss')
            if log_scale:
                plt.yscale('log')
        if save==True:
            plt.savefig(f'figs/mbloss_{k}_{run}.pdf', format="pdf", bbox_inches="tight")



        # plot errors
        if plot_error:
            methods_e = (ae2,ae,ae3)
            plt.figure(figsize=(20,5))
            colors_ = [ 'r','b', 'y','g']  # , 'g', 'c', 'm', 'k']
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



        plt.tight_layout()
        if save==True and plot_error:
            plt.savefig(f'figs/testerror_{k}_{run}.pdf', format="pdf", bbox_inches="tight")
        #plt.savefig('tikzpicture_plots/fig_27.pdf', format="pdf", bbox_inches="tight")
        plt.show()



def plot_loss_error_avg(k,li_epoch ,plot_error=True, log_scale=True, no_inits = 10, save = False):
    plot_grads = False
    run="0"
    if True:
        with open(f"results_data_spirals/Exp{k}_1.json") as file: #abs max
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
            print(f'shape of a {a.shape}')

        with open(f"results_data_spirals/Exp{k}_2.json") as file: #abs min

            a_temp = [f[i]['losses'] for i in f.keys()]
            at2 = f[run]['times'] 
            if plot_error:
                a_err = [f[i]['errors'] for i in f.keys()]
                
            
            a2 = np.array(a_temp)
            if plot_error:
                ae2 = np.array(a_err)
            print(f'shape of a {a.shape}')



        with open(f"results_data_spirals/Exp{k}_3.json") as file: # baseline
            f = json.load(file)

            a_temp = [f[i]['losses'] for i in f.keys()]
            at3 = f[run]['times'] 
            if plot_error:
                a_err = [f[i]['errors'] for i in f.keys()]
                
            
            a3 = np.array(a_temp)
            if plot_error:
                ae3 = np.array(a_err)
            print(f'shape of a {a3.shape}')


        

        labels = [ 'LImin','LI','baseline']
        colors = ['b', 'r', 'y','g']  # , 'g', 'c', 'm', 'k']

        methods = (a2,a)#,a3)
        times = (at2,at)#,at3)


        matplotlib.rc('ytick', labelsize=12)
        matplotlib.rc('xtick', labelsize=12)
        fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [5,5]})
        fig.set_size_inches((20,10))

        # first subplot plot losses
        colors_ = [ 'r','b', 'y','g']  # , 'g', 'c', 'm', 'k']
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
            axes[0].vlines(li_epoch*10, 0, 1, colors='gray', linestyles='dashed')
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
            methods_e = (ae2,ae)#,ae3)
            colors_ = [ 'r','b', 'y','g']  # , 'g', 'c', 'm', 'k']
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
                axes[1].vlines(li_epoch, 0, 60, colors='gray', linestyles='dashed')
                axes[1].legend()
                #plt.ylim([0, 100])
                #ma = max([a.shape[1] for a in methods])
                # plt.xlim([0, ma])
                axes[1].set_xlabel('epochs')
                axes[1].set_ylabel('test error')
            #plt.show()



        fig.tight_layout()
        if save==True and plot_error:
            fig.savefig(f'figs/testerror_{k}_avg.pdf', format="pdf", bbox_inches="tight")
        #plt.savefig('tikzpicture_plots/fig_27.pdf', format="pdf", bbox_inches="tight")
        plt.show()


plot_loss_error_avg(k,li_epoch, plot_error=True, log_scale=True, no_inits = 10, save = False)

#for run in ["0","1","2","3","4","5","6","7","8","9"]:
#    plot_loss_error_single_init(k,li_epoch ,plot_error=True, log_scale=True, run=run, save = True)
    