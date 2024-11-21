import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib

from smoothers import moving_average

fileno = 80
run = "0" # check also others!
moving_avg = 1000

loss_all = []
error_all = []
times_all=[]

labels=['1',"10","20", "30", "40","50","60","70","80", 'baseline']
no_exps = 9

for k in range(0,no_exps+1):

    with open(f"results_data_spirals/Exp{fileno}_{k}.json") as file:
        # a_temp = list(json.load(file).values())
        f = json.load(file)
        a_temp = [f[run]['losses']]
        a_temp = moving_average(a_temp[0], moving_avg)
        a = np.array(a_temp)
        
        
    loss_all.append(a)

for k in range(0,no_exps+1):

    with open(f"results_data_spirals/Exp{fileno}_{k}.json") as file:
        # a_temp = list(json.load(file).values())
        f = json.load(file)
        aa_temp = [f[run]['errors']]
        aa = np.array(aa_temp)
        #print(aa[0,1])
        
    error_all.append(aa)

for k in range(0,no_exps+1):

    with open(f"results_data_spirals/Exp{fileno}_{k}.json") as file:
        # a_temp = list(json.load(file).values())
        f = json.load(file)
        at_temp = [f[run]['times']]
        at = np.array(at_temp)
        
    times_all.append(at)


# uncomment when run9 exitst
#with open(f"results_data_spirals/Exp{fileno}_9.json") as file:
#    f=json.load(file)
#    g = [f[run]['grad_norms']]

#print(g)

matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('xtick', labelsize=16)
#plt.figure(figsize=(20,5))
fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [5,5]})
fig.set_size_inches((20,10))

##### plot loss###################
i=0
li_pt = 6000
for a, t in zip(loss_all, times_all):
    len_t = t[0,1:].shape[0]

    #print(len_t)
    if True:
        axes[0].plot(a, '-', label=labels[i], linewidth=2)
    if i < 8:
        axes[0].vlines(li_pt,0.02,0.7,linestyles='dotted',colors='gray')
    if i==9:
        axes[0].vlines(600,0.02,0.7,linestyles='dotted',colors='gray')
    i+=1
    li_pt+= 6000

axes[0].set_yscale('log')
axes[0].set_xlabel('iterations', fontsize=20)
axes[0].set_ylabel('loss', fontsize=20)
axes[0].legend(fontsize=15,loc=3)

########plot test error###################
li_pt = 6000
i=0
for a, t in zip(error_all, times_all):
    len_t = t[0,1:].shape[0]

    print(len_t)
    if True:
        #plt.plot(a[0,:], 'o', label=labels[i], markersize=2)
        axes[1].plot(a[0,:], label=labels[i], linewidth=2)
    if i < 8:
        axes[1].vlines(li_pt,0,100,linestyles='dotted',colors='gray')
    i+=1
    li_pt+= 6000


axes[1].set_xlabel('iterations', fontsize=20)
axes[1].set_ylabel('test error (%)', fontsize=20)
axes[1].set_ylim(top=100,bottom=0)
#plt.xlim(left=70, right=140)
axes[1].legend(fontsize=15)
plt.title(f'abs max')

plt.tight_layout()
plt.savefig('../../Papers/plots/when-to-insert-fnns-loss-and-error_MNIST.pdf', format="pdf", bbox_inches="tight")
plt.show()
