import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib

fileno = 263#262#261
run = "0"

save = None#'plots/when-to-insert-fnns-loss-and-error-mb.pdf'

loss_all = []
error_all = []
times_all=[]

no_batches = 10

labels=["LI50","LI100","LI150","LI200","LI250","LI300","LI350","LI400","FNN1","FNNlarge"]
no_exps = 9 # 10 for large FNN

for k in range(1,no_exps+1):

    with open(f"results_data_spirals/Exp{fileno}_{k}.json") as file:
        # a_temp = list(json.load(file).values())
        f = json.load(file)
        a_temp = [f[run]['losses']]
        a = np.array(a_temp)
        
    loss_all.append(a)

for k in range(1,no_exps+1):

    with open(f"results_data_spirals/Exp{fileno}_{k}.json") as file:
        # a_temp = list(json.load(file).values())
        f = json.load(file)
        aa_temp = [f[run]['errors']]
        aa = np.array(aa_temp)
        print(f'aa shape {aa.shape}')
        
    error_all.append(aa)

for k in range(1,no_exps+1):

    with open(f"results_data_spirals/Exp{fileno}_{k}.json") as file:
        # a_temp = list(json.load(file).values())
        f = json.load(file)
        at_temp = [f[run]['times']]
        at = np.array(at_temp)
        
    times_all.append(at)



with open(f"results_data_spirals/Exp{fileno}_9.json") as file:
    f=json.load(file)
    g = [f[run]['grad_norms']]

#print(g)

matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('xtick', labelsize=16)
#plt.figure(figsize=(20,5))
fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [5,5]})
fig.set_size_inches((20,10))


i=0
li_pt = 50*no_batches
for a, t in zip(loss_all, times_all):
    len_t = t[0,1:].shape[0]

    print(len_t)
    if True:
        axes[0].plot(a[0,:], '-', label=labels[i], linewidth=2)
    if i < 8:
        axes[0].vlines(li_pt,0.009,0.8,linestyles='dotted',colors='blue')
    i+=1
    li_pt+= 50*no_batches

axes[0].set_yscale('log')
axes[0].set_xlabel('iterations', fontsize=20)
axes[0].set_ylabel('(minibatch) loss', fontsize=20)
axes[0].legend(fontsize=15,loc='lower left')


li_pt = 50*no_batches
i=0
for a, t in zip(error_all, times_all):
    len_t = t[0,1:].shape[0]
    mean1 = np.nanmean(a, axis=0)
    # only every tenth entry in mean1 is not Nan
    mean1 = mean1[::10]
    # generate suitbel x axis list
    x = np.arange(0,len(mean1)*10,10)

    print(len_t)
    if True:
        print(mean1[1])
        #plt.plot(a[0,:], 'o', label=labels[i], markersize=2)
        axes[1].plot(x,mean1,'-', label=labels[i], linewidth=3)
    if i < 8:
        axes[1].vlines(li_pt,0,70,linestyles='dotted',colors='blue')
    i+=1
    li_pt+= 50*no_batches


axes[1].set_xlabel('iterations', fontsize=20)
axes[1].set_ylabel('test error (%)', fontsize=20)
#axes[1].set_ylim(top=61,bottom=0)
#axes[1].set_xlim(left=0, right=5500)
axes[1].legend(fontsize=15,loc='lower left')
#plt.title(f'{i}')

plt.tight_layout()
if save is not None:
    plt.savefig(save, format="pdf", bbox_inches="tight")
#plt.savefig('../../Papers/plots/when-to-insert-fnns-loss-and-error.pdf', format="pdf", bbox_inches="tight")
plt.show()
