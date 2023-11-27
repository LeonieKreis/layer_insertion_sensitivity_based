import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib

fileno = 88
run = "0"

loss_all = []
error_all = []
times_all=[]

labels=["150","250","350","450","550","650","750","850","FNN1","FNN2"]
no_exps = 9

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
li_pt = 150
for a, t in zip(loss_all, times_all):
    len_t = t[0,1:].shape[0]

    print(len_t)
    if True:
        axes[0].plot(a[0,:], '-', label=labels[i], linewidth=2)
    if i < 8:
        axes[0].vlines(li_pt,0.2,0.7,linestyles='dotted',colors='gray')
    i+=1
    li_pt+= 100

axes[0].set_yscale('log')
axes[0].set_xlabel('iterations', fontsize=20)
axes[0].set_ylabel('loss', fontsize=20)
axes[0].legend(fontsize=15,loc=3)


li_pt = 150
i=0
for a, t in zip(error_all, times_all):
    len_t = t[0,1:].shape[0]

    print(len_t)
    if True:
        #plt.plot(a[0,:], 'o', label=labels[i], markersize=2)
        axes[1].plot(a[0,:], label=labels[i], linewidth=2)
    if i < 8:
        axes[1].vlines(li_pt,0,70,linestyles='dotted',colors='gray')
    i+=1
    li_pt+= 100


axes[1].set_xlabel('iterations', fontsize=20)
axes[1].set_ylabel('test error (%)', fontsize=20)
axes[1].set_ylim(top=70,bottom=0)
#plt.xlim(left=70, right=140)
axes[1].legend(fontsize=15,loc=3)
#plt.title(f'{i}')

plt.tight_layout()
plt.savefig('../../Papers/plots/when-to-insert-fnns-loss-and-error.pdf', format="pdf", bbox_inches="tight")
plt.show()
