import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os

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

def gen_gif_of_heatmaps(list_vals, num, vmin, vmax,path, noepochs, nobatches):
    epoch_curr = -1
    batch_curr = 0
    for i, val in enumerate(list_vals):
        if i%nobatches == 0:
            batch_curr = 0
            epoch_curr += 1
        else:
            batch_curr += 1
        sns.heatmap(val, vmin=vmin, vmax=vmax)
        plt.title(f'Grads: epoch {epoch_curr}, batch {batch_curr}, param {num}')
        plt.savefig(f'{path}_{epoch_curr}_{batch_curr}_{num}.png', format="png", bbox_inches="tight")        
        plt.close()

    
################# main function ############################

def weights_to_heatmapgif(txtpath, noepochs, nobatches, param_num, plotpath=None, delete = True):
    '''
    txtpath should be of the form {} as in '{heatmaps/heatmapvals_}epoch{e}_batch{b}_param{p}.txt'
    plotpath should be of the form {} as in '{heatmaps/heatmapvals/}'
    '''
    if plotpath is None:
        plotpath = txtpath

    for num in range(param_num):
        list_all = []
        img_array = []
        for e in range(noepochs):
            for b in range(nobatches):
                vals_curr = load_weights(txtpath,e,b,num)
                if len(vals_curr.shape)==1:
                    vals_curr = vals_curr.reshape(vals_curr.shape[0],1)
                list_all.append(vals_curr)
        # set vmin as min of all values and vmax as max of all values in list_all
        vmin = min([vals.min() for vals in list_all])
        vmax = max([vals.max() for vals in list_all])
        gen_gif_of_heatmaps(list_all,num,  vmin, vmax, plotpath, noepochs, nobatches)
        #for filename in sorted(glob.glob(f'{plotpath}iteration*_num{num}.png')):
        for e in range(noepochs):
            for b in range(nobatches):
                filename = f'{plotpath}_{e}_{b}_{num}.png'
                img = Image.open(filename)
                img_array.append(img)
        img_array[0].save(f'{plotpath}param_{num}.gif', save_all=True, append_images=img_array[1:], duration=200, loop=0)
        print(f'Movie for param {num} saved')
        if delete:
            for e in range(noepochs):
                for b in range(nobatches):
                    filename = f'{plotpath}_{e}_{b}_{num}.png'
                    os.remove(filename)
            print(f'files for param {num} deleted')
    
        

############## make movie out of the heatmaps for each weight (=num) separate ##########
import os
import numpy as np
import glob
from PIL import Image



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
    
    img_array[0].save(f'{path}param_{num}.gif', save_all=True, append_images=img_array[1:], duration=200, loop=0)
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