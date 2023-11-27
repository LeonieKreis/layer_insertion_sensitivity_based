import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 9

run = "0"

plot_error=False
plot_grads = False
log_scale = True

for i in ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29"]:
    i="29"
    with open(f"results_data_spirals/Exp{k}_1.json") as file:
        f = json.load(file)
        a_temp = f[i]['losses'] 
        at = f[run]['times'] 
        if plot_error:
            a_err = [f[i]['errors'] for i in f.keys()]
        if plot_grads:
            a_grad = f[run]['grad_norms']
        a = np.array(a_temp)
        if plot_error:
            ae = np.array(a_err)
        print(f'shape of a {a.shape}')


    with open(f"results_data_spirals/Exp{k}_3.json") as file:
        f = json.load(file)
        c_temp = f[i]['losses'] 
        ct = f[run]['times'] 
        if plot_error:
            c_err = [f[i]['errors'] for i in f.keys()]
        if plot_grads:
            c_grad = f[run]['grad_norms']
        c = np.array(c_temp)
        if plot_error:
            ce = np.array(c_err)
        print(f'shape of c {c.shape}')
        print(len(ct))

    with open(f"results_data_spirals/Exp{k}_4.json") as file:
        f = json.load(file)
        d_temp = f[i]['losses'] 
        dt = f[run]['times'] 
        if plot_error:
            d_err = [f[i]['errors'] for i in f.keys()]
        if plot_grads:
            d_grad = f[run]['grad_norms']
        d = np.array(d_temp)
        if plot_error:
            de = np.array(d_err)
        print(f'shape of d {d.shape}')
        print(len(dt))

    labels = ['LI','N1','N2']

    methods = (a,c,d)
    times = (at,ct,dt)


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
        plt.plot(ta[1:end_weg],mean1, colors_[i], label=label)
        plt.legend()
        #plt.ylim([0, 2])
        #ma = max([a.shape[1] for a in methods])
        #plt.xlim([0, ma])
        plt.xlabel('time(s)')
        plt.ylabel(' (fullbatch) loss')
        if log_scale:
            plt.yscale('log')



    # plot errors
    if plot_error:
        methods_e = (ae,ce,de)
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
            plt.plot(ta[0:end_weg], mean1, colors_[i] + 'o', label=label,
                        markersize=5)  # , linestyle='o')
        
            plt.legend()
            plt.ylim([0, 100])
            #ma = max([a.shape[1] for a in methods])
            # plt.xlim([0, ma])
            plt.xlabel('time(s)')
            plt.ylabel('test error')
        #plt.show()



    plt.tight_layout()
    #plt.savefig('tikzpicture_plots/fig_27.pdf', format="pdf", bbox_inches="tight")
    plt.show()

