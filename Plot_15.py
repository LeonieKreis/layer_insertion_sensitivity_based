import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 151#00

#run = "3" #"9"
#i=run

plot_error=True
plot_grads = True
log_scale = True

save = False

def plot_loss_error_single_init(plot_error, plot_grads, log_scale, run,k, save = False):
    i=run
    avg = 0.9 # 0.9
    if True:
        with open(f"results_data_spirals/Exp{k}_1.json") as file: #abs max
            f = json.load(file)
            a_temp = f[i]['losses'] 
            at = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = f[i]['errors']
            if plot_grads:
                a_grad = f[run]['grad_norms']
            a = np.array(a_temp)
            if plot_error:
                ae = np.array(a_err)
            print(f'shape of a {a.shape}')

        with open(f"results_data_spirals/Exp{k}_2.json") as file: # abs min
            f = json.load(file)
            a_temp = f[i]['losses'] 
            at2 = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = f[i]['errors']
            
            a2 = np.array(a_temp)
            if plot_error:
                ae2 = np.array(a_err)
            print(f'shape of a {a2.shape}')

        with open(f"results_data_spirals/Exp{k}_3.json") as file: # baseline
            f = json.load(file)
            a_temp = f[i]['losses'] 
            at3 = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = f[i]['errors']
            
            a3 = np.array(a_temp)
            if plot_error:
                ae3 = np.array(a_err)
            print(f'shape of a {a3.shape}')


        with open(f"results_data_spirals/Exp{k}_4.json") as file: # baseline
            f = json.load(file)
            a_temp = f[i]['losses'] 
            at4 = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = f[i]['errors']
            
            a4 = np.array(a_temp)
            if plot_error:
                ae4 = np.array(a_err)
            print(f'shape of a {a4.shape}')
    

        labels = ['LI', 'baseline']#['LI','LI min', 'baseline','big']

        methods = (a,a3)#(a,a2,a3,a4)
        times = (at,at3)#(at,at2,at3,at4)


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
            if avg is not None:
                mean1 = ema2_np(mean1, avg)
            plt.plot(mean1, colors_[i], label=label)
            plt.vlines(450*10, 0, 1, colors='gray', linestyles='dashed')
            plt.legend()
            #plt.ylim([0, 2])
            #ma = max([a.shape[1] for a in methods])
            #plt.xlim([0, ma])
            plt.xlabel('its')
            plt.ylabel(' (minibatch) loss')
            if log_scale:
                plt.yscale('log')
        if save==True:
            plt.savefig(f'figs/mbloss_{k}_{run}.pdf', format="pdf", bbox_inches="tight")



        # plot errors
        if plot_error:
            methods_e = (ae,ae3)#(ae,ae2,ae3,ae4)
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
                plt.vlines(450, 0, 100, colors='gray', linestyles='dashed')
                plt.legend()
                plt.ylim([0, 100])
                #ma = max([a.shape[1] for a in methods])
                # plt.xlim([0, ma])
                plt.xlabel('epochs')
                plt.ylabel('test error')
            #plt.show()

        if plot_grads:  # limax
            only_weights = True
            labels0 = ['W1','b1', 'W2']
            labels1 = ['W1','b1','W2','b2','W3']
            # optional:remove biases from grad structure?
            
            plt.figure(figsize=(20,5))
            for i, pg in enumerate(a_grad[0]):
                if avg is not None:
                    pg = ema2_np(pg, avg)
                if only_weights and i%2==1:
                    continue
                plt.plot(pg, label=labels0[i])
            no_its0 = len(a_grad[0][0])
            no_its1 = len(a_grad[1][0])
            for i,pg in enumerate(a_grad[1]):
                if avg is not None:
                    pg = ema2_np(pg, avg)
                if only_weights and i%2==1:
                    continue
                plt.plot(range(no_its0, no_its0 + no_its1),pg, label=labels1[i])
            plt.legend()
            plt.yscale('log')
            plt.xlabel('iterations')
            plt.ylabel('weight grad norms')
            



        plt.tight_layout()
        if save==True and plot_error:
            plt.savefig(f'figs/testerror_{k}_{run}.pdf', format="pdf", bbox_inches="tight")
        #plt.savefig('tikzpicture_plots/fig_27.pdf', format="pdf", bbox_inches="tight")
        plt.show()
        
for run in ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38"]:
    print(run)
    plot_loss_error_single_init(plot_error, plot_grads, log_scale, run,k, save)