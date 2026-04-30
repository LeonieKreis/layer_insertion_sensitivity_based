import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 112
# 112 (115-118)



plot_error=True
plot_grads = False
log_scale = True



def plot_loss_error_mean(plot_error, plot_grads, log_scale, run,k, save = False):
    
    if True:
        with open(f"results_data_spirals/Exp{k}_1.json") as file: #abs max
            f = json.load(file)
            a_temp = [f[i]['losses'] for i in f.keys()] 
            for i in range(len(a_temp)):
                if len(a_temp[i])!=5000: print('incorrect lentgh at',i)
                a_temp[i]=a_temp[i]+list(np.zeros(5000-len(a_temp[i])))
            
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
                if len(a_temp[i])!=5000: print('incorrect lentgh at',i)
                a_temp[i]=a_temp[i]+list(np.zeros(5000-len(a_temp[i])))
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
            for i in range(len(a_temp)):
                if len(a_temp[i])!=5000: print('incorrect lentgh at',i)
                a_temp[i]=a_temp[i]+list(np.zeros(5000-len(a_temp[i])))
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


        with open(f"results_data_spirals/Exp{k}_4.json") as file: # big resnet
            f = json.load(file)
            a_temp = [f[i]['losses'] for i in f.keys()]
            for i in range(len(a_temp)):
                if len(a_temp[i])!=5000: print('incorrect lentgh at',i)
                a_temp[i]=a_temp[i]+list(np.zeros(5000-len(a_temp[i])))
            
            at4 = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = [f[i]['errors'] for i in f.keys()]
                
                for i in range(10):
                    if len(a_err[i])!=5000: print('incorrect lentgh at',i)
            if plot_grads:
                a_grad = f[run]['grad_norms']
            a4 = np.array(a_temp)
            if plot_error:
                ae4 = np.array(a_err)
            print(f'shape of a {a4.shape}')

        with open(f"results_data_spirals/Exp{k}_5.json") as file: # random
            f = json.load(file)
            a_temp = [f[i]['losses'] for i in f.keys()] 
            for i in range(len(a_temp)):
                if len(a_temp[i])!=5000: print('incorrect lentgh at',i)
                a_temp[i]=a_temp[i]+list(np.zeros(5000-len(a_temp[i])))
            
            at5 = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err5 = [f[i]['errors'] for i in f.keys()]
            if plot_grads:
                a_grad = f[run]['grad_norms']
            a5 = np.array(a_temp)
            if plot_error:
                ae5 = np.array(a_err5)
            print(f'shape of a5 {a5.shape}')

    

        labels = ['SensLI','ResNet1','ResNet2']#'Random']#['SensLI','LIother']##['LI','LIother', 'ResNet1','ResNet2','random']

        methods = (a,a3,a4)#(a,a2)#(a,a2,a3,a4,a5)
        times = (at,at3,at4)#(at,at2)#(at,at2,at3,at4,at5)


        # plt.figure(figsize=(20,5))

        # # first subplot plot losses
        # colors_ = ['b', 'r', 'y','g','orange']  # , 'g', 'c', 'm', 'k']
        # for i, (aa, ta) in enumerate(zip(methods, times)):
        #     print(aa.shape)
        #     mean1 = np.nanmean(aa, axis=0)
        #     #mean1=aa
        #     if labels is None:
        #         label = str(i)
        #     else:
        #         label = labels[i]
        #     if i==0:
        #         end_weg = -1
        #     else:
        #         end_weg = None
        #     #plt.plot(ta[1:end_weg],mean1, colors_[i], label=label)
        #     plt.plot(mean1, colors_[i], label=label)
        #     plt.vlines(100*10, 0, 1, colors='blue', linestyles='dotted')
        #     plt.vlines(200*10, 0, 1, colors='blue', linestyles='dotted')
        #     plt.vlines(300*10, 0, 1, colors='blue', linestyles='dotted')
        #     plt.legend()
        #     #plt.ylim([0, 2])
        #     #ma = max([a.shape[1] for a in methods])
        #     #plt.xlim([0, ma])
        #     plt.xlabel('iterations')
        #     plt.ylabel(' (minibatch) loss')
        #     if log_scale:
        #         plt.yscale('log')
        # if save==True:
        #     plt.savefig(f'figs/mbloss_{k}_{run}.pdf', format="pdf", bbox_inches="tight")



        # # plot errors
        # if plot_error:
        #     methods_e = (ae,ae2,ae3,ae4,ae5)
        #     plt.figure(figsize=(20,5))
        #     colors_ = ['b', 'r', 'y','g','orange']  # , 'g', 'c', 'm', 'k']
        #     for i, (aa, ta) in enumerate(zip(methods_e, times)):
        #         #print(aa.shape)
        #         mean1 = np.nanmean(aa, axis=0)
        #         #mean1=aa

        #         if labels is None:
        #             label = str(i)
        #         else:
        #             label = labels[i]
        #         if i==0:
        #             end_weg = -1
        #         else:
        #             end_weg = None
        #         #plt.plot(ta[0:end_weg], mean1, colors_[i] + 'o', label=label,
        #         #            markersize=5)  # , linestyle='o')
        #         plt.plot( mean1, colors_[i] , label=label,
        #                     markersize=5)  # , linestyle='o')
        #         plt.vlines(100, 0, 100, colors='blue', linestyles='dotted')
        #         plt.vlines(200, 0, 100, colors='blue', linestyles='dotted')
        #         plt.vlines(300, 0, 100, colors='blue', linestyles='dotted')
        #         plt.legend()
        #         plt.ylim([0, 100])
        #         #ma = max([a.shape[1] for a in methods])
        #         # plt.xlim([0, ma])
        #         plt.xlabel('epochs')
        #         plt.ylabel('test error')
        #     #plt.show()

        # rewrite the commented text such that the plots are in two subplots insetad of separate
        if not plot_error:
            matplotlib.rc('ytick', labelsize=18)
            matplotlib.rc('xtick', labelsize=18)
            colors_ = ['b','orange']
            plt.figure(figsize=(20,5))
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
                plt.plot(mean1, colors_[i], label=label)
                plt.vlines(100*10, 0, 1, colors='blue', linestyles='dotted')
                plt.vlines(200*10, 0, 1, colors='blue', linestyles='dotted')
                plt.vlines(300*10, 0, 1, colors='blue', linestyles='dotted')
                plt.legend(fontsize=20)
                #plt.ylim([0, 2])
                #ma = max([a.shape[1] for a in methods])
                #plt.xlim([0, ma])
                plt.xlabel('iterations',fontsize=20)
                plt.ylabel(' (minibatch) loss', fontsize=20)
                if log_scale:
                    plt.yscale('log')


        else:
            matplotlib.rc('ytick', labelsize=18)
            matplotlib.rc('xtick', labelsize=18)
            fig, ax = plt.subplots(2, 1, figsize=(20,10))
            # first subplot plot losses
            colors_ = ['b', 'r', 'g','y','orange']
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
                ax[0].plot(mean1, colors_[i], label=label)
                ax[0].vlines(100*10, 0, 1, colors='blue', linestyles='dotted')
                ax[0].vlines(200*10, 0, 1, colors='blue', linestyles='dotted')
                ax[0].vlines(300*10, 0, 1, colors='blue', linestyles='dotted')
                ax[0].legend(fontsize=20)
                #plt.ylim([0, 2])
                #ma = max([a.shape[1] for a in methods])
                #plt.xlim([0, ma])
                ax[0].set_xlabel('iterations',fontsize=20)
                ax[0].set_ylabel(' (minibatch) loss', fontsize=20)
                if log_scale:
                    ax[0].set_yscale('log')
                
            # plot errors
            if plot_error:
                methods_e = (ae,ae3,ae4)#(ae,ae2,ae3,ae4,ae5)
                #plt.figure(figsize=(20,5))
                colors_ = ['b', 'r', 'g','y','orange']
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
                    ax[1].plot( mean1, colors_[i] , label=label,
                                markersize=5)
                    ax[1].vlines(100, 0, 100, colors='blue', linestyles='dotted')
                    ax[1].vlines(200, 0, 100, colors='blue', linestyles='dotted')
                    ax[1].vlines(300, 0, 100, colors='blue', linestyles='dotted')
                    ax[1].legend(fontsize=20)
                    ax[1].set_ylim([0, 100])    
                    ax[1].set_xlabel('epochs',fontsize=20)
                    ax[1].set_ylabel('test error',fontsize=20)  


        plt.tight_layout()
        if save==True:
            plt.savefig(f'figs/comp-fixed-arch-resnets-3lis-mb.pdf', format="pdf", bbox_inches="tight")
        #plt.savefig('tikzpicture_plots/fig_27.pdf', format="pdf", bbox_inches="tight")
        plt.show()
        
save = False#True
run = "0" # only for grad
plot_error=True
plot_grads = False
plot_loss_error_mean(plot_error, plot_grads, log_scale, run,k, save)