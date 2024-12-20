import sys 

sys.path.append('../layer_insertion_sensitivity_based')

from heatmap import weights_to_heatmapgif,weights_to_heatmapgif_in_memory, delete_png_files, weight_to_pic_heatmap

k=23   

if False:
    grads=True
    vals = not grads
    if grads:
        string = 'grads'
        string2=string
    if vals:
        string = 'vals'
        string2 = 'weights'
    k=1
    txtpath=f'heatmaps/{string2}/Exp23/li_max/{string}_{k}/'
    plotpath=f'heatmaps/gifs/Exp23/li_max/{string}_{k}'
    txtpath2=f'heatmaps/{string2}/Exp23/baseline/{string}_{k}/'
    

    no_epochs=100
    no_batches=10
    param_it = [0,2,4,6]
    param_it2 = [0,2,4]

    weights_to_heatmapgif(txtpath, no_epochs, no_batches, param_it, plotpath=plotpath,delete=True, log=True, grads=True, txtpath2=txtpath2, param_it2=param_it2)
    

for nobatch in [0,1,2]:
    param_num = 4
    no_epoch = 0
    grads= True
    if grads:
        string = 'grads'
        string2=string
    if not grads:
        string = 'vals'
        string2 = 'weights'
    kk=1
    txtpath=f'heatmaps/{string2}/Exp{k}/li_max/{string}_{kk}/'
    plotpath=f'heatmaps/gifs/Exp{k}/li_max/{string}_{kk}'


    weight_to_pic_heatmap(txtpath,param_num, nobatch, no_epoch, path=plotpath, log=True)