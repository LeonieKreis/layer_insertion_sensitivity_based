import heatmap

# old code
# epoch = 0

# param_num=2
# for batch in [0,1,2]:
#     txtpath = f'heatmaps/grads/grads_cnn/grads_'
#     plotpath = f'heatmaps/gifs/cnn_mb/limax/'


#     heatmap.grads_to_pic_heatmap_pdf(txtpath,param_num, batch, epoch, log=True, path=plotpath )



FB = False
MB = False #not FB
old_fnnfb = True#False
old_fnnmb = True#False

################## FB comp ############################################################################
if FB:
    arc='fnn'

    repo_id = arc+'_fb_new'

    txtpath = f'heatmaps/grads/grads_{repo_id}/'
    plotpath = f'heatmaps/gifs/{repo_id}/limax/'

    if arc=='res2': list_params_relevant = [2,4,5,7]
    if arc=='fnn': list_params_relevant = [2,4]

    for epoch in [0,1,2,9]:

        for param_num in list_params_relevant:
            batch = 0
            heatmap.grads_to_pic_heatmap_pdf(txtpath,param_num, batch, epoch, log=True, path=plotpath )

################ MB COMP ############################################################################
if MB:
    arc='res2'

    repo_id = arc+'_mb_new'


    txtpath = f'heatmaps/grads/grads_{repo_id}/'
    plotpath = f'heatmaps/gifs/{repo_id}/limax/'

    if arc=='res2': list_params_relevant = [2,4,5,7]
    if arc=='fnn': list_params_relevant = [2,4]

    for epoch in [0,1,2,9]:
        for batch in [0,1,2]:
            for param_num in list_params_relevant:
                heatmap.grads_to_pic_heatmap_pdf(txtpath,param_num, batch, epoch, log=True, path=plotpath )


#######################################################################################################

if old_fnnmb:
    print('old fnn mb')
    arc='fnn'
    txtpath = f'heatmaps/grads/Exp23/li_max/grads_1/'
    plotpath = f'heatmaps/gifs/Exp23/li_max/'

    epoch = 0
    for param_num in [4]: 
        for batch in [0,1,2]:
            heatmap.grads_to_pic_heatmap_pdf(txtpath,param_num, batch, epoch, log=True, path=plotpath )

if old_fnnfb:
    print('old fnn fb')
    arc='fnn'
    txtpath = f'heatmaps/grads/Exp24/li_max/grads_1/'
    plotpath = f'heatmaps/gifs/Exp24/li_max/'
    batch = 0

    for param_num in [4]:
        for epoch in [0,1,2]:
            heatmap.grads_to_pic_heatmap_pdf(txtpath,param_num, batch, epoch, log=True, path=plotpath )

    