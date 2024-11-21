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

def gen_gif_of_heatmaps(list_vals, num, vmin, vmax,path):
    for i, val in enumerate(list_vals):
        sns.heatmap(val, vmin=vmin, vmax=vmax)
        plt.title(f'Grads: iteration {i}, weight {num}')
        plt.savefig(f'{path}iteration_{i}_num{num}.png', format="png", bbox_inches="tight")        
        plt.close()

    
################# main function ############################

def weights_to_heatmapgif(txtpath, noepochs, nobatches, param_num, vmin, vmax, plotpath=None):
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
                list_all.append(vals_curr)
        gen_gif_of_heatmaps(list_all,num,  vmin, vmax, plotpath)
        for filename in sorted(glob.glob(f'{plotpath}iteration*_num{num}.png')):
            img = Image.open(filename)
            img_array.append(img)
        img_array[0].save(f'{plotpath}param_{num}.gif', save_all=True, append_images=img_array[1:], duration=200, loop=0)
        print(f'Movie for param {num} saved')
    
        

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