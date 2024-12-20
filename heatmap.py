import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import glob
from PIL import Image
from matplotlib.colors import LogNorm
import io

def save_weightgrads_heatmap(model, batch, e,path='heatmaps/heatmapgrads'):
    '''
    Saves a heatmap of the gradients of the weights in each step.
    
    Args:
        model: (nn.Module) model
        batch: (int) batch number
        e: (int) epoch number
        path: (str) path to save the heatmaps

    '''
    with torch.no_grad():
        for num,p in enumerate(model.parameters()):
                values = p.grad.data.numpy()
                if num%2 == 1:
                    values = values.reshape(values.shape[0],1)
                sns.heatmap(values, vmin=-3, vmax=3)
                plt.title(f'Grads: epoch {e}, batch {batch}, weight {num}')
                plt.savefig(f'{path}_{e}_{batch}_{num}.png', format="png", bbox_inches="tight")
                
                plt.close()

def save_weightvalues_heatmap(model, batch, e,path='heatmaps/heatmapvals'):
    '''
    Saves a heatmap of the gradients of the weights in each step.
    
    Args:
        model: (nn.Module) model
        batch: (int) batch number
        e: (int) epoch number
        path: (str) path to save the heatmaps

    '''
    with torch.no_grad():
        for num,p in enumerate(model.parameters()):
                values = p.data.numpy()
                if num%2 == 1:
                    values = values.reshape(values.shape[0],1)
                sns.heatmap(values, vmin=-3, vmax=3)
                plt.title(f'Vals: epoch {e}, batch {batch}, weight {num}')
                plt.savefig(f'{path}_{e}_{batch}_{num}.png', format="png", bbox_inches="tight")
                
                plt.close()

def load_weights(txtpath, epoch, batch, param_num):
    '''
    Load weights from a text file.
    
    Args:
        weightspath: (str) path to the weights directory
        epoch: (int) epoch number
        batch: (int) batch number
        param_num: (int) parameter number
    
    Returns:
        weights: (numpy array) loaded weights
    '''
    filename = f'{txtpath}epoch{epoch}_batch{batch}_param{param_num}.txt'
    weights = np.loadtxt(filename)
    return weights

def weight_to_pic_heatmap(txtpath, param_num,nobatch, noepoch, log=True, path='heatmaps/heatmapvals'):
    weight = load_weights(txtpath,noepoch,nobatch,param_num)
    if len(weight.shape)==1:
        weight = weight.reshape(weight.shape[0],1)
    weight = np.abs(weight)
    vmin = weight.min()
    if vmin==0:
        vmin = 1e-5
    vmax = weight.max()
    if vmax == 0:
        vmax = 1e-5
    
    if log:
        sns.heatmap(weight, norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        sns.heatmap(weight, vmin=vmin, vmax=vmax)
    plt.title(f'Vals: epoch {noepoch}, batch {nobatch}')
    plt.savefig(f'{path}_{noepoch}_{nobatch}_{param_num}.png', format="png", bbox_inches="tight")
    plt.close()

    
def gen_gif_of_heatmaps(list_vals, param_it, vmin, vmax,path, nobatches, log=False, grads=True, list_all2=None, param_it2=None):
    if grads: title='Grads'
    else: title='Vals'
    
    if list_all2 is None:
        epoch_curr = -1
        batch_curr = 0
        for i, val in enumerate(list_vals):
            if i%nobatches == 0:
                batch_curr = 0
                epoch_curr += 1
            else:
                batch_curr += 1
            no_axes = len(list(param_it))
            fig, axs = plt.subplots(nrows=1, ncols=no_axes, figsize=(3*no_axes,3))
            plt.suptitle(f'{title}: epoch {epoch_curr}, batch {batch_curr}')
            for j, num in enumerate(param_it):
                if log:
                    sns.heatmap(val[j], ax=axs[j], norm=LogNorm(vmin=vmin, vmax=vmax))
                else:
                    sns.heatmap(val[j], vmin=vmin, vmax=vmax, ax=axs[j])
                    
            fig.savefig(f'{path}_{epoch_curr}_{batch_curr}.png', format="png", bbox_inches="tight")        
            plt.close()

    else:
        epoch_curr = -1
        batch_curr = 0 
        for i, (val, val2) in enumerate(zip(list_vals, list_all2)):
            if i%nobatches == 0:
                batch_curr = 0
                epoch_curr += 1
            else:
                batch_curr += 1
            no_axes = len(list(param_it))
            fig, axs = plt.subplots(nrows=2, ncols=no_axes, figsize=(3*no_axes,2*3))
            plt.suptitle(f'{title}: epoch {epoch_curr}, batch {batch_curr}')
            for j, num in enumerate(param_it):
                if log:
                    sns.heatmap(val[j], ax=axs[0,j], norm=LogNorm(vmin=vmin, vmax=vmax))
                else:
                    sns.heatmap(val[j], vmin=vmin, vmax=vmax, ax=axs[0,j])
            for j, num in enumerate(param_it2):
                if log:
                    sns.heatmap(val2[j], ax=axs[1,j], norm=LogNorm(vmin=vmin, vmax=vmax))
                else:
                    sns.heatmap(val2[j], vmin=vmin, vmax=vmax, ax=axs[1,j])
            
            fig.savefig(f'{path}_{epoch_curr}_{batch_curr}.png', format="png", bbox_inches="tight")        
            plt.close()

    
################# main function ############################

def weights_to_heatmapgif(txtpath, noepochs, nobatches, param_it, plotpath=None, delete = True, log=False, grads=True, txtpath2=None, param_it2=None):
    '''
    txtpath should be of the form {} as in '{heatmaps/heatmapvals_}epoch{e}_batch{b}_param{p}.txt'
    plotpath should be of the form {} as in '{heatmaps/heatmapvals/}'
    '''


    if plotpath is None:
        plotpath = txtpath
    
    

    if txtpath2 is None:
        list_all = []
        img_array = []
        for e in range(noepochs):
            for b in range(nobatches):
                list_param = []
                for num in param_it:
                    vals_curr = load_weights(txtpath,e,b,num)
                    if log:
                        # turn values into absolute values
                        vals_curr = np.abs(vals_curr)
                    if len(vals_curr.shape)==1:
                        vals_curr = vals_curr.reshape(vals_curr.shape[0],1)
                    list_param.append(vals_curr)
                list_all.append(list_param)
        # set vmin as min of all values and vmax as max of all values in list_all
        vmin = min([vals.min() for sublist in list_all for vals in sublist])
        if vmin == 0:
            vmin = 1e-5
        vmax = max([vals.max() for sublist in list_all for vals in sublist])
        gen_gif_of_heatmaps(list_all,param_it,  vmin, vmax, plotpath, nobatches, log=log, grads=grads)
    
        for e in range(noepochs):
            for b in range(nobatches):
                filename = f'{plotpath}_{e}_{b}.png'
                img = Image.open(filename)
                img_array.append(img)
        img_array[0].save(f'{plotpath}.gif', save_all=True, append_images=img_array[1:], duration=20, loop=3,optimize=True)
        print(f'Movie for param_it saved')
        if delete:
            for e in range(noepochs):
                for b in range(nobatches):
                    filename = f'{plotpath}_{e}_{b}.png'
                    os.remove(filename)
            print(f'files for param_it deleted')

    else:
        list_all = []
        img_array = []
        for e in range(noepochs):
            for b in range(nobatches):
                list_param = []
                for num in param_it:
                    vals_curr = load_weights(txtpath,e,b,num)
                    if log:
                        # turn values into absolute values
                        vals_curr = np.abs(vals_curr)
                    if len(vals_curr.shape)==1:
                        vals_curr = vals_curr.reshape(vals_curr.shape[0],1)
                    list_param.append(vals_curr)
                list_all.append(list_param)
        # set vmin as min of all values and vmax as max of all values in list_all
        vmin = min([vals.min() for sublist in list_all for vals in sublist])
        if vmin == 0:
            vmin = 1e-5
        vmax = max([vals.max() for sublist in list_all for vals in sublist])

        list_all2 = []
        img_array2 = []
        for e in range(noepochs):
            for b in range(nobatches):
                list_param = []
                for num in param_it2:
                    vals_curr = load_weights(txtpath2,e,b,num)
                    if log:
                        # turn values into absolute values
                        vals_curr = np.abs(vals_curr)
                    if len(vals_curr.shape)==1:
                        vals_curr = vals_curr.reshape(vals_curr.shape[0],1)
                    list_param.append(vals_curr)
                list_all2.append(list_param)
        # set vmin as min of all values and vmax as max of all values in list_all2
        vmin2 = min([vals.min() for sublist in list_all2 for vals in sublist])
        if vmin2 == 0:
            vmin2 = 1e-5
        vmax2 = max([vals.max() for sublist in list_all2 for vals in sublist])

        vmin = min(vmin,vmin2)
        vmax = max(vmax,vmax2)

        gen_gif_of_heatmaps(list_all,param_it,  vmin, vmax, plotpath, nobatches, log=log, grads=grads, list_all2=list_all2, param_it2=param_it2)

        for e in range(noepochs):
            for b in range(nobatches):
                filename = f'{plotpath}_{e}_{b}.png'
                img = Image.open(filename)
                img_array.append(img)
        img_array[0].save(f'{plotpath}.gif', save_all=True, append_images=img_array[1:], duration=20, loop=3,optimize=True)
        print(f'Movie for param_it saved')
        if delete:
            for e in range(noepochs):
                for b in range(nobatches):
                    filename = f'{plotpath}_{e}_{b}.png'
                    os.remove(filename)
            print(f'files for param_it deleted')

    print('All gifs saved')

def delete_png_files(noepochs_it, nobatches, num, plotpath):
    if num is not None:
        for e in noepochs_it:
                for b in range(nobatches):
                    filename = f'{plotpath}_{e}_{b}_{num}.png'
                    os.remove(filename)
    else:
        for e in noepochs_it:
                for b in range(nobatches):
                    filename = f'{plotpath}_{e}_{b}.png'
                    os.remove(filename)

    print(f'files for param {num} deleted')
    
############## in memory #####################
def gen_gif_of_heatmaps_in_memory(list_vals, param_it, vmin, vmax, nobatches, log=False):
    '''
    Generate heatmaps and save them as images in memory.
    
    Args:
        list_vals: (list) list of weight values
        param_it: (iterable) parameter indices
        vmin: (float) minimum value for color scale
        vmax: (float) maximum value for color scale
        nobatches: (int) number of batches
        log: (bool) whether to use logarithmic color scale
    
    Returns:
        img_array: (list) list of images in memory
    '''
    img_array = []
    epoch_curr = -1
    batch_curr = 0
    for i, val in enumerate(list_vals):
        if i % nobatches == 0:
            batch_curr = 0
            epoch_curr += 1
        else:
            batch_curr += 1
        no_axes = len(list(param_it))
        fig, axs = plt.subplots(nrows=1, ncols=no_axes, figsize=(10, 6))
        for j, num in enumerate(param_it):
            if log:
                sns.heatmap(val[j], ax=axs[j], norm=LogNorm(vmin=vmin, vmax=vmax))
            else:
                sns.heatmap(val[j], vmin=vmin, vmax=vmax, ax=axs[j])
        plt.suptitle(f'Grads: epoch {epoch_curr}, batch {batch_curr}')
        
        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Load the image from the BytesIO object
        buf.seek(0)
        img = Image.open(buf)
        img_array.append(img)
        buf.close()
    
    return img_array

def weights_to_heatmapgif_in_memory(txtpath, noepochs, nobatches, param_it, plotpath=None, log=False):
    '''
    Generate GIFs from weight values without saving intermediate images.
    
    Args:
        txtpath: (str) path to the text files with weight values
        noepochs: (int) number of epochs
        nobatches: (int) number of batches
        param_it: (iterable) parameter indices to process
        plotpath: (str) path to save the GIFs
        log: (bool) whether to use logarithmic color scale
    '''
    if plotpath is None:
        plotpath = txtpath

    list_all = []
    for e in range(noepochs):
        for b in range(nobatches):
            list_param = []
            for num in param_it:
                vals_curr = load_weights(txtpath, e, b, num)
                if log:
                    # turn values into absolute values
                    vals_curr = np.abs(vals_curr)
                if len(vals_curr.shape) == 1:
                    vals_curr = vals_curr.reshape(vals_curr.shape[0], 1)
                list_param.append(vals_curr)
            list_all.append(list_param)
    
    # Calculate vmin and vmax across all parameters, epochs, and batches
    vmin = min([vals.min() for sublist in list_all for vals in sublist])
    if vmin == 0:
        vmin = 1e-5
    vmax = max([vals.max() for sublist in list_all for vals in sublist])
    
    img_array = gen_gif_of_heatmaps_in_memory(list_all, param_it, vmin, vmax, nobatches, log=log)
    
    img_array[0].save(f'{plotpath}.gif', save_all=True, append_images=img_array[1:], duration=20, loop=3, optimize=True)
    print(f'Movie for param_it saved')      

############## make movie out of the heatmaps for each weight (=num) separate ##########




def make_movie(path, num, no_epochs=None, no_batches=None):
    '''
    Makes a movie out of the heatmaps for each weight (=num) separate.
    
    Args:
        path: (str) path to the heatmaps
        num: (int) weight number
    '''
    img_array = []
    if no_epochs is None and no_batches is None:
        for filename in sorted(glob.glob(f'{path}*_{num}.png')):
            img = Image.open(filename)
            img_array.append(img)

    else:
        for e in range(no_epochs):
            for b in range(no_batches):
                filename = f'{path}_{e}_{b}_{num}.png'
                img = Image.open(filename)
                img_array.append(img)
    
    img_array[0].save(f'{path}param_{num}.gif', save_all=True, append_images=img_array[1:], duration=20, loop=3)
    print(f'Movie for param {num} saved')





if __name__=='__main__':

    from nets import feed_forward
    from stepfunction_dataset import gen_steps_dataset

    model = feed_forward(1,1,1,3,flatten=False)
    train, test, x,y = gen_steps_dataset(batchsize=37)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i, (x,y) in enumerate(train):
        model.zero_grad()
        loss = torch.nn.CrossEntropyLoss()( model(x),y)
        loss.backward()
        #for p in model.parameters(): 
        #    print(p.data.detach().numpy())
        #    print(p.grad.data.detach().numpy())
        #save_weightgrads_heatmap(model, 1, i)
        #save_weightvalues_heatmap(model, 1, i)
        print(i)
        optimizer.step()
        if i==0: break
    # pathgrads = 'heatmaps/heatmapgrads'
    # pathvals = 'heatmaps/heatmapvals'
    # for num in range(3):  
    #     make_movie(pathgrads, num)
    #     make_movie(pathvals, num)

# save in txt
# load from txt
# make heatmap
# make movie