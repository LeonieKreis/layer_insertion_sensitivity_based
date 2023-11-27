import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib

fileno = 8
run = "0"

loss_all = []
error_all = []
labels=["150","250","350","450","550","650","750","850","FNN1","FNN2"]

for k in range(1,11):

    with open(f"results_data_spirals/Exp{fileno}_{k}.json") as file:
        # a_temp = list(json.load(file).values())
        f = json.load(file)
        a_temp = [f[run]['losses']]
        a = np.array(a_temp)
        
    loss_all.append(a)

for k in range(1,11):

    with open(f"results_data_spirals/Exp{fileno}_{k}.json") as file:
        # a_temp = list(json.load(file).values())
        f = json.load(file)
        aa_temp = [f[run]['errors']]
        aa = np.array(aa_temp)
        
    error_all.append(aa)



with open(f"results_data_spirals/Exp{fileno}_9.json") as file:
    f=json.load(file)
    g = [f[run]['grad_norms']]

#print(g)

matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('xtick', labelsize=16)
plt.figure(figsize=(20,5))

i=0
li_pt = 150
for a in loss_all:
    plt.plot(a[0,:], '--', label=labels[i], linewidth=2)
    if i < 8:
        plt.vlines(li_pt,min(a[0,:]),max(a[0,:]),linestyles='dotted',colors='gray')
    i+=1
    li_pt+= 100

plt.yscale('log')
plt.xlabel('iterations', fontsize=20)
plt.ylabel('loss', fontsize=20)
#plt.ylim()
#plt.xlim(left=70, right=140)
plt.legend(fontsize=20)
#plt.title(f'{i}')


plt.tight_layout()
#plt.savefig('tikzpicture_plots/fig_27.pdf', format="pdf", bbox_inches="tight")
#plt.show()


matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('xtick', labelsize=16)
plt.figure(figsize=(20,5))

i=0
li_pt = 150
for a in error_all:
    plt.plot(a[0,:], 'o', label=labels[i], linewidth=1)
    if i < 8:
        plt.vlines(li_pt,min(a[0,:]),max(a[0,:]),linestyles='dotted',colors='gray')
    i+=1
    li_pt+= 100


plt.xlabel('iterations', fontsize=20)
plt.ylabel('test error', fontsize=20)
#plt.ylim()
#plt.xlim(left=70, right=140)
plt.legend(fontsize=20)
#plt.title(f'{i}')

# plot grads of original net
plt.figure(figsize=(20, 5))
grad_norms2 = g
l1 = len(grad_norms2[0])
print(l1)
for i in range(l1):
    plt.plot(grad_norms2[0][i], label=f'{i}')
plt.legend()
plt.yscale('log')
plt.xlabel('iterations')
plt.ylabel('layerwise gradient norms scaled by lr')
plt.title('RN1')
plt.show()