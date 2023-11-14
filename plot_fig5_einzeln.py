import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib
import torch
import plot_helper
from utils import ema_np, ema2_np

k = 15 #'14alt' #9 for a or 14 for b

line = 20
# bei 13 ist gut : 8
# bei 9 ist gut : 14


labels = ['Layer insertion', 'Alternative']

with open(f"results_data/Exp{k}_1.json") as file:
    # a_temp = list(json.load(file).values())
    f = json.load(file)
    a_temp = [f[i]['losses'] for i in f.keys()]
    a = np.array(a_temp)
    
    # print(f'shape of a {a.shape}')

if True:
    with open(f"results_data/Exp{k}_2.json") as file: #Exp8_3
        # b_temp = list(json.load(file).values())
        f = json.load(file)
        b_temp = [f[i]['losses'] for i in f.keys()]
        
        b = np.array(b_temp)
        




matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('xtick', labelsize=16)
#plt.figure(figsize=(20,5))

for i in range(0,30):
    print('i')
    plt.figure(figsize=(20,5))

    #mean1 = np.nanmean(a, axis=0)
    mean1 = list(a[i,:])
    print(len(mean1))
    mean1.pop(line)
    print(len(mean1))
    plt.plot(mean1, '-', label=labels[0])

    mean2 = list(b[i,:])
    print(len(mean2))
    mean2.pop(line)
    print(len(mean2))
    #mean2 = np.nanmean(b, axis=0)
    plt.plot(mean2, '--', label=labels[1])



    plt.vlines(line,min(mean1),max(mean1),linestyles='dotted',colors='gray')

    plt.yscale('log')
    plt.xlabel('iterations', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    #plt.ylim()
    #plt.xlim(left=70, right=140)
    plt.legend(fontsize=20)
    plt.title(f'{i}')


    plt.tight_layout()
    #plt.savefig('tikzpicture_plots/fig4.pdf', format="pdf", bbox_inches="tight")
    plt.show()