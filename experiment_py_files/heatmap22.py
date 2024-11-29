import sys 

sys.path.append('../layer_insertion_sensitivity_based')

from heatmap import weights_to_heatmapgif,weights_to_heatmapgif_in_memory, delete_png_files

if False:
    for i in [2]:
        for k in range(1,2):
            txtpath=f'heatmaps/grads/Exp21/li_max/grads_{i}_{k}/'
            plotpath=f'heatmaps/gifs/Exp21/li_max/grads_{i}_{k}'

            txtpath2=f'heatmaps/weights/Exp21/li_max/vals_{i}_{k}/'
            plotpath2=f'heatmaps/gifs/Exp21/li_max/vals_{i}_{k}'

            noepochs=500#450 + k*950
            nobatches=10
            param_it=[2]


            weights_to_heatmapgif(txtpath, noepochs, nobatches, param_it, plotpath=plotpath,delete = True)
            weights_to_heatmapgif(txtpath2, noepochs, nobatches, param_it, plotpath=plotpath2,delete = True)


if False:
    delete_png_files(range(100,500), nobatches=10, num=None, plotpath='heatmaps/gifs/Exp21/li_max/grads_0_1')
    

if True:
    grads=True
    vals = not grads
    if grads:
        string = 'grads'
        string2=string
    if vals:
        string = 'vals'
        string2 = 'weights'
    k=1
    txtpath=f'heatmaps/{string2}/Exp22/li_max/{string}_{k}/'
    plotpath=f'heatmaps/gifs/Exp22/li_max/{string}_{k}'
    txtpath2=f'heatmaps/{string2}/Exp22/baseline/{string}_{k}/'
    

    no_epochs=200
    no_batches=10
    param_it = [0,2,4,6]
    param_it2 = [0,2,4]

    weights_to_heatmapgif(txtpath, no_epochs, no_batches, param_it, plotpath=plotpath,delete=True, log=True, grads=True, txtpath2=txtpath2, param_it2=param_it2)
    
