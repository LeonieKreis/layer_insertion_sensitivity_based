import sys 

sys.path.append('../layer_insertion_sensitivity_based')

from heatmap import weights_to_heatmapgif



for k in range(6):
    txtpath=f'heatmaps/grads/exp3/li_max/grads_{k}/'
    plotpath=f'heatmaps/gifs/exp3/li_max/grads_{k}'

    txtpath2=f'heatmaps/weights/exp3/li_max/vals_{k}/'
    plotpath2=f'heatmaps/gifs/exp3/li_max/vals_{k}'

    noepochs=500
    nobatches=1
    param_num=3+2*k


    weights_to_heatmapgif(txtpath, noepochs, nobatches, param_num, plotpath=plotpath,delete = True)
    weights_to_heatmapgif(txtpath2, noepochs, nobatches, param_num, plotpath=plotpath2,delete = True)