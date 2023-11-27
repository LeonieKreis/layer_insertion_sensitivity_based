#  ########## necessary imports ###############################################
import torch
import random
import os
import copy
import sys
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

sys.path.append('../layer_insertion_sensitivity_based')

from layer_insertion_loop import layer_insertion_loop
from train_and_test_ import train, check_testerror
from nets import feed_forward, two_weight_resnet, one_weight_resnet
from save_to_json import write_losses
from spirals_data_new import gen_spiral_dataset, plot_decision_boundary

# ################# fix hyperparameters ###################################

# for checking the progress of the training in the terminal, use the bash command: jp length filename.json
# to see how many runs are already saved

k = 0000

# seed
s=1
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)

torch.set_num_threads(8)

# Define hyperparameters

hidden_layers = 2
fix_width = 3
epochs_classical = 4000
wanted_testerror = 0.
_type = 'res2'
act_fun = nn.Tanh
interval_testerror = 1

batchsize = 450 # fullbatch
no_per_class = 300
r0=0.5
circles = 1


td, vd, data_X, data_y = gen_spiral_dataset(batchsize,no_per_class,r0,circles)


no_steps_per_epoch = 1



save_grad_norms= True

lr_init = 1e-1
optimizer_type = 'SGD'

lrscheduler_type = 'StepLR'


# for classical

lr_init_classical = lr_init
lrscheduler_args_classical = {'step_size': 400000,
                              'gamma': 0.1}




dim_in = 2
dim_out = 2

# classical net small
kwargs_net_classical2 = {
    'hidden_layers': hidden_layers,
    'dim_hidden_layers': fix_width,
    'act_fun': act_fun,
    'type': _type
}


max_length = epochs_classical

# determine which trainings are run
T1 = True
T2 = True
T3 = True
T4 = True

# define no of training run instances

no_of_initializations = 1  

# declare path where json files are saved

path1 = f'results_data_spirals/Exp{k}.json'
if os.path.isfile(path1):
    print(f' file with path {path1} already exists!')
    quit()


for i in range(no_of_initializations):

# build net for classical big
    if _type == 'fwd':
        model_classical2 = feed_forward(dim_in, dim_out, **kwargs_net_classical2)
    if _type == 'res2':
        model_classical2 = two_weight_resnet(
            dim_in, dim_out, **kwargs_net_classical2)
    if _type == 'res1':
        model_classical2 = one_weight_resnet(
            dim_in, dim_out, **kwargs_net_classical2)
        
    

    # build optimizer and lr scheduler for the classical training:
    # build optimizer
    if optimizer_type == 'SGD':
        optimizer_classical2 = torch.optim.SGD(
            model_classical2.parameters(), lr_init_classical)

    # build lr scheduler
    if lrscheduler_type == 'StepLR':
        step_size2 = lrscheduler_args_classical['step_size']
        gamma2 = lrscheduler_args_classical['gamma']
        lrscheduler_classical2 = torch.optim.lr_scheduler.StepLR(
            optimizer_classical2, step_size=step_size2, gamma=gamma2)

    
    print('classical training big!')
    if T4:
        print('training classically on model', model_classical2)
        mblosses_classical2, lr_end2, test_error_classical2, exit_flag4, grad_norm4, times4 = train(model_classical2,
                                                                 train_dataloader=td,
                                                                 epochs=epochs_classical,
                                                                 optimizer=optimizer_classical2,
                                                                 scheduler=lrscheduler_classical2,
                                                                 wanted_testerror=wanted_testerror,
                                                                 start_with_backtracking=None,
                                                                 check_testerror_between=interval_testerror,
                                                                 test_dataloader=vd,
                                                                 print_param_flag=False,
                                                                 save_grad_norms=save_grad_norms
                                                                 )
        
        plot_decision_boundary(model_classical2, data_X, data_y)

        # save losses3
        write_losses(f'results_data_spirals/Exp{k}_4.json', mblosses_classical2, max_length, structures=[
                    epochs_classical], errors=test_error_classical2,
                    interval_testerror=interval_testerror, exit_flag=exit_flag4,
                    grad_norms= grad_norm4,
                    its_per_epoch=no_steps_per_epoch)

        # full_list_of_losses_3.append(mb_losses_comp)
        
        print(f'test error of classical training big {test_error_classical2[-1]}')

        plt.plot(times4[1:], mblosses_classical2)
        plt.ylabel('loss')
        plt.xlabel('time (s)')
        plt.show()
        plt.plot(times4[1:], test_error_classical2,'o')
        plt.ylabel('test error')
        plt.xlabel('time (s)')
        plt.show()
    # next step in loop
