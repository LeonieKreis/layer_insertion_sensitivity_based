import heatmap


epoch = 0

param_num=2
for batch in [0,1,2]:
    txtpath = f'heatmaps/grads/grads_cnn/grads_'
    plotpath = f'heatmaps/gifs/cnn_mb/limax/'


    heatmap.weight_to_pic_heatmap(txtpath,param_num, batch, epoch, log=True, path=plotpath )