import numpy as np
import json
import matplotlib.pyplot as plt
import torch


k = 3

plot_error=True
plot_grads = True
log_scale = True

save = False

def plot_loss_error_single_init(plot_error, plot_grads, log_scale, run,k, save = False):
    i=run
    if True:
        with open(f"results_data_steps/Exp{k}_1.json") as file: #abs max
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

        with open(f"results_data_steps/Exp{k}_2.json") as file: # abs min
            f = json.load(file)
            a_temp = f[i]['losses'] 
            at2 = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = f[i]['errors']
            if plot_grads:
                a2_grad = f[run]['grad_norms']
            a2 = np.array(a_temp)
            if plot_error:
                ae2 = np.array(a_err)
            print(f'shape of a {a2.shape}')

        with open(f"results_data_steps/Exp{k}_3.json") as file: # baseline
            f = json.load(file)
            a_temp = f[i]['losses'] 
            at3 = f[run]['times'] 
            if plot_error:
                #a_err = [f[i]['errors'] for i in f.keys()]
                a_err = f[i]['errors']
            if plot_grads:
                a3_grad = f[run]['grad_norms']
            a3 = np.array(a_temp)
            if plot_error:
                ae3 = np.array(a_err)
            print(f'shape of a {a3.shape}')



        labels = ['LI','LI min', 'baseline']

        methods = (a,a2,a3)
        times = (at,at2,at3)


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
            plt.vlines(500*1, 0, 1, colors='gray', linestyles='dashed')
            plt.vlines(2*500*1, 0, 1, colors='gray', linestyles='dashed')
            plt.vlines(3*500*1, 0, 1, colors='gray', linestyles='dashed')
            plt.vlines(4*500*1, 0, 1, colors='gray', linestyles='dashed')
            plt.vlines(5*500*1, 0, 1, colors='gray', linestyles='dashed')
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
            methods_e = (ae,ae2,ae3)
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
                plt.plot( mean1, colors_[i] , label=label,
                            markersize=5)  # , linestyle='o')
                plt.vlines(500, 0, 100, colors='gray', linestyles='dashed')
                plt.vlines(2*500, 0, 100, colors='gray', linestyles='dashed')
                plt.vlines(3*500, 0, 100, colors='gray', linestyles='dashed')
                plt.vlines(4*500, 0, 100, colors='gray', linestyles='dashed')
                plt.vlines(5*500, 0, 100, colors='gray', linestyles='dashed')
                plt.legend()
                plt.yscale('log')
                #ma = max([a.shape[1] for a in methods])
                # plt.xlim([0, ma])
                plt.xlabel('epochs')
                plt.ylabel('test error')
            #plt.show()

        # plot grads of 
        if plot_grads:
            plt.figure(figsize=(20,5))
            # print(len(a_grad))
            # print(len(a_grad[0]))
            # print(len(a_grad[1]))
            # print(len(a_grad[2]))
            # print(len(a_grad[3]))
            # print(len(a_grad[4]))
            # print(len(a_grad[5]))
            # print(len(a_grad[0][0]))
            # print(len(a_grad[0][1]))
            start_x = 0
            for l in range(len(a_grad)):
                curr = a_grad[l]
                for p in range(len(curr)):
                    plt.plot(list(range(start_x,start_x+len(curr[p]))),curr[p], label=f'layer{p}{l}')
                start_x += len(curr[0])
            plt.legend(ncol=l+1)
            plt.yscale('log')
            plt.xlabel('its')
            plt.ylabel('grad norms paramwise')
            

        plt.tight_layout()
        if save==True and plot_error:
            plt.savefig(f'figs/testerror_{k}_{run}.pdf', format="pdf", bbox_inches="tight")
        #plt.savefig('tikzpicture_plots/fig_27.pdf', format="pdf", bbox_inches="tight")
        plt.show()

for run in ["0"]:#,"1","2","3","4","5","6","7","8","9"]:
    plot_loss_error_single_init(plot_error, plot_grads, log_scale, run,k,save)