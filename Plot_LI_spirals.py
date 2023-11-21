import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib

fileno = 12
run = "0"

loss_all = []
error_all = []
times_all=[]

labels=["150","250","350","450","550","650","750","850","RN1","RN2"]
no_exps = 10

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
plt.figure(figsize=(20,5))

i=0
li_pt = 150
for a, t in zip(loss_all, times_all):
    print(t.shape)
    if i<8:
        plt.plot(t[0,1:-1],a[0,:], '--', label=labels[i], linewidth=2)
    else:
        if fileno==10 and i==9:
            plt.plot(t[0,1:],a[0,:-38], '--', label=labels[i], linewidth=2)
        else:
            plt.plot(t[0,1:],a[0,:], '--', label=labels[i], linewidth=2)
    
    #if i < 8:
    #    plt.vlines(li_pt,min(a[0,:]),max(a[0,:]),linestyles='dotted',colors='gray')
    i+=1
    li_pt+= 100

plt.yscale('log')
plt.xlabel('time(s)', fontsize=20)
plt.ylabel('loss', fontsize=20)
#plt.ylim()
#plt.xlim(left=70, right=140)
plt.legend(fontsize=20)
#plt.title(f'{i}')


plt.tight_layout()
#plt.savefig('tikzpicture_plots/fig_27.pdf', format="pdf", bbox_inches="tight")
plt.show()
