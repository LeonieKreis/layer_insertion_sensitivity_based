#  ########## necessary imports ###############################################
import torch
import random
import os
import copy
import sys
import numpy as np
from torch import nn

sys.path.append('../layer_insertion_sensitivity_based')

from layer_insertion_loop import layer_insertion_loop
from train_and_test_ import train, check_testerror
from nets import feed_forward, two_weight_resnet, one_weight_resnet
from save_to_json import write_losses
from spirals_data_new import gen_spiral_dataset

# ################# fix hyperparameters ###################################

# for checking the progress of the training in the terminal, use the bash command: jp length filename.json
# to see how many runs are already saved

k = 0

# seed
s=1
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)

torch.set_num_threads(8)

# Define hyperparameters

hidden_layers_start = 2
fix_width = [3,3]
no_iters = 1
lr_decrease_after_li = 1.
epochs = [0,1]  # [10, 5, 5]
wanted_testerror = 0.
_type = 'res2'
act_fun = nn.Tanh
interval_testerror = 1

batchsize = 45 #450 # fullbatch
no_per_class = 300
r0=0.5
circles = 1


td, vd, data_X, data_y = gen_spiral_dataset(batchsize,no_per_class,r0,circles)


no_steps_per_epoch = 1
print('no of iterations in one epoch:', no_steps_per_epoch)

end_list = []
for i, e in enumerate(epochs):
    end_list.append(e)
    end_list.append(1)
end_list.pop()  # removes last 1 which was too much




save_grad_norms= True

lr_init = 1e-1
optimizer_type = 'SGD'
lrscheduler_type = 'StepLR'
lrscheduler_args = {'step_size': 4000,
                    'gamma': 0.1}


# for classical
epochs_classical = sum(epochs)
lr_init_classical = lr_init
lrscheduler_args_classical = {'step_size': 4000,
                              'gamma': 0.1}
hidden_layers_classical = hidden_layers_start#no_iters + hidden_layers_start
fix_width_classical = fix_width

# build models:

kwargs_net = {
    'hidden_layers': hidden_layers_start,
    'dim_hidden_layers': fix_width,
    'act_fun': act_fun,
    'type': _type
}

dim_in = 2
dim_out = 2


# classical net small
kwargs_net_classical = {
    'hidden_layers': hidden_layers_classical,
    'dim_hidden_layers': fix_width_classical,
    'act_fun': act_fun,
    'type': _type
}

# classical net small
kwargs_net_classical2 = {
    'hidden_layers': hidden_layers_classical+1,
    'dim_hidden_layers': 2*fix_width_classical,
    'act_fun': act_fun,
    'type': _type
}


max_length = max(epochs_classical, sum(epochs))*no_steps_per_epoch

# determine which trainings are run
T1 = True
T2 = True
T3 = True
T4 = True

# define no of training run instances

no_of_initializations = 1  # 50

# set up empty lists for saving the observed quantities
# (besides the save to the json file)

final_testerror1 = []
final_testerror2 = []
final_testerror3 = []
final_testerror4 = []

# declare path where json files are saved

path1 = f'results_data_spirals/Exp{k}_1.json'
if os.path.isfile(path1):
    print(f' file with path {path1} already exists!')
    quit()


for i in range(no_of_initializations):
    print(f'loop number {i}!')

    # build net for ali 1
    # build model
    if _type == 'fwd':
        model_init = feed_forward(dim_in, dim_out, **kwargs_net)
    if _type == 'res2':
        model_init = two_weight_resnet(dim_in, dim_out, **kwargs_net)
    if _type == 'res1':
        model_init = one_weight_resnet(dim_in, dim_out, **kwargs_net)

    param_init = torch.nn.utils.parameters_to_vector(model_init.parameters())
    param_init2 = copy.deepcopy(param_init.data)
    param_init_class = copy.deepcopy(param_init.data)

    # train ali 1
    print('training on first ali')
    if T1:
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
            save_grad_norms=save_grad_norms,
            save_heatmaps=True
        )
        
        print(grad_norm1)
        # save losses1 and final accuracy1
        write_losses(path1,
                     mb_losses1, max_length, end_list, test_errors1,
                     interval_testerror=interval_testerror, times=times1,grad_norms = grad_norm1, its_per_epoch=no_steps_per_epoch)    # save losses3

        # full_list_of_losses_1.append(mb_losses)
        final_testerror1.append(test_errors_short1[-1])

