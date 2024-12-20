#  ########## necessary imports ###############################################
import torch
import random
import os
import copy
import sys
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#sys.path.append('../layer_insertion_sensitivity_based')
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Add the constructed path to the Python path
sys.path.append(parent_dir)

from layer_insertion_loop import layer_insertion_loop
from train_and_test_ import train, check_testerror
from nets import feed_forward, two_weight_resnet, one_weight_resnet
from save_to_json import write_losses


# ################# fix hyperparameters ###################################

# for checking the progress of the training in the terminal, use the bash command: jp length filename.json
# to see how many runs are already saved

k = 'test'

save_data = False

# seed
s=1
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)

torch.set_num_threads(8)

# Define hyperparameters

hidden_layers_start = 1 # number of hidden layers
fix_width = [50] # widths of hidden layer
no_iters = 2 # number of layer insertions
lr_decrease_after_li = 1.0
epochs = [5,5,5]  
wanted_testerror = 0.
_type = 'fwd' #'fwd' or 'res2
act_fun = nn.ReLU
interval_testerror = 1
save_grad_norms= True

batchsize = 100 # minibatch

# Get MNIST dataset
MNIST=True
if MNIST:    
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    tr_split_len = 60000#10000
    te_split_len = 10000#int(0.33*tr_split_len)
    training_data = torch.utils.data.random_split(training_data, [tr_split_len, len(training_data)-tr_split_len])[0]
    test_data = torch.utils.data.random_split(test_data, [te_split_len, len(test_data)-te_split_len])[0]

print('no of iterations in one epoch:',int(len(training_data)/batchsize))
print(len(training_data))
print(len(test_data))

td = DataLoader(training_data, batch_size=batchsize)
vd = DataLoader(test_data, batch_size=10000)


###################################################################################################################

no_steps_per_epoch = 600
print('no of iterations in one epoch:', no_steps_per_epoch)

end_list = []
for i, e in enumerate(epochs):
    end_list.append(e)
    end_list.append(1)
end_list.pop()  # removes last 1 which was too much




lr_init = 0.01 #0.05
optimizer_type = 'Adam' # 'Adam' or 'SGD'
lrscheduler_type = 'StepLR' # only 'StepLR' implemented
lrscheduler_args = {'step_size': 5,
                    'gamma': 0.5}


# for classical
epochs_classical = sum(epochs)
lr_init_classical = lr_init
lrscheduler_args_classical = {'step_size': 5,
                              'gamma': 0.5}
hidden_layers_classical = hidden_layers_start
fix_width_classical = fix_width 

# build models:

kwargs_net = { # baseline for layer insertion
    'hidden_layers': hidden_layers_start,
    'dim_hidden_layers': fix_width,
    'act_fun': act_fun,
    'type': _type
}


dim_in = 28*28
dim_out = 10



# classical net small
kwargs_net_classical = {
    'hidden_layers': hidden_layers_classical,
    'dim_hidden_layers': fix_width_classical,
    'act_fun': act_fun,
    'type': _type
}



max_length = max(epochs_classical, sum(epochs))*no_steps_per_epoch

# determine which trainings are run
T1 = True
T2 = True

# define no of training run instances

no_of_initializations = 1 


# declare path where json files are saved

path1 = f'results_data_spirals/Exp{k}_1.json'
if os.path.isfile(path1):
    print(f' file with path {path1} already exists!')
    quit()


for i in range(no_of_initializations):
    print(f'loop number {i}!')

    # build net for classical 
    if _type == 'fwd':
        model_classical = feed_forward(dim_in, dim_out, **kwargs_net_classical)
    if _type == 'res2':
        model_classical = two_weight_resnet(
            dim_in, dim_out, **kwargs_net_classical)
    if _type == 'res1':
        model_classical = one_weight_resnet(
            dim_in, dim_out, **kwargs_net_classical)
        

    # build optimizer and lr scheduler for the classical training:
    # build optimizer
    if optimizer_type == 'SGD':
        optimizer_classical = torch.optim.SGD(
            model_classical.parameters(), lr_init_classical)
    if optimizer_type == 'Adam':
        optimizer_classical = torch.optim.Adam(
            model_classical.parameters(), lr_init_classical, weight_decay=5e-4)

    # build lr scheduler
    if lrscheduler_type == 'StepLR':
        step_size = lrscheduler_args_classical['step_size']
        gamma = lrscheduler_args_classical['gamma']
        lrscheduler_classical = torch.optim.lr_scheduler.StepLR(
            optimizer_classical, step_size=step_size, gamma=gamma)
        



    param_init = torch.nn.utils.parameters_to_vector(model_classical.parameters())
    param_init2 = copy.deepcopy(param_init.data)
    
    

    # train classical  small
    print('classical training small!')
    if T1:
        print('training classically on model', model_classical)
        mblosses_classical, lr_end, test_error_classical, exit_flag3, grad_norm3, times3 = train(model_classical,
                                                                 train_dataloader=td,
                                                                 epochs=epochs_classical,
                                                                 optimizer=optimizer_classical,
                                                                 scheduler=lrscheduler_classical,
                                                                 wanted_testerror=wanted_testerror,
                                                                 start_with_backtracking=None,
                                                                 check_testerror_between=interval_testerror,
                                                                 test_dataloader=vd,
                                                                 print_param_flag=False,
                                                                 save_grad_norms=save_grad_norms
                                                                 )
        if save_data:
            # save losses3
            write_losses(f'results_data_spirals/Exp{k}_3.json', mblosses_classical, max_length, structures=[
                        epochs_classical], errors=test_error_classical,
                        interval_testerror=interval_testerror, times=times3, grad_norms = grad_norm3,
                        its_per_epoch=no_steps_per_epoch)

    # build net for ali 1
    # build model
    if _type == 'fwd':
        model_init = feed_forward(dim_in, dim_out, **kwargs_net)
    if _type == 'res2':
        model_init = two_weight_resnet(dim_in, dim_out, **kwargs_net)
    if _type == 'res1':
        model_init = one_weight_resnet(dim_in, dim_out, **kwargs_net)
    
    torch.nn.utils.vector_to_parameters(param_init2, model_init.parameters())

    # train ali 1
    print('training on first ali')
    if T2:
        model1, mb_losses1, test_errors_short1, test_errors1, exit_flag1, grad_norm1, times1, sens1 = layer_insertion_loop(
            iters=no_iters,
            epochs=epochs,
            model=model_init,
            kwargs_net=kwargs_net,
            dim_in=dim_in,
            dim_out=dim_out,
            train_dataloader=td,
            test_dataloader=vd,
            lr_init=lr_init,
            wanted_test_error=wanted_testerror,
            mode='abs max',
            optimizer_type=optimizer_type,
            lrschedule_type=lrscheduler_type,
            lrscheduler_args=lrscheduler_args,
            check_testerror_between=interval_testerror,
            decrease_after_li=lr_decrease_after_li,
            print_param_flag=False,
            start_with_backtracking=None,
            v2=False,
            save_grad_norms=save_grad_norms
        )
        
        if save_data:
            # save losses1 and final accuracy1
            write_losses(path1,
                        mb_losses1, max_length, end_list, test_errors1,
                        interval_testerror=interval_testerror, times=times1,grad_norms = grad_norm1, its_per_epoch=no_steps_per_epoch, sens=sens1)    # save losses3

