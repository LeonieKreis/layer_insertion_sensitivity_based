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

sys.path.append('../layer_insertion_sensitivity_based')

from layer_insertion_loop import layer_insertion_loop
from train_and_test_ import train, check_testerror
from nets import feed_forward, two_weight_resnet, one_weight_resnet
from save_to_json import write_losses

# ################# fix hyperparameters ###################################

# for checking the progress of the training in the terminal, use the bash command: jp length filename.json
# to see how many runs are already saved

k = 28

# seed
s=2
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)

torch.set_num_threads(8)

# Define hyperparameters

hidden_layers_start = 3
fix_width = 70
no_iters = 1
lr_decrease_after_li = 1.
epochs = [10,50+70]  # [10, 5, 5]
wanted_testerror = 1.
_type = 'res2'
act_fun = nn.Tanh
interval_testerror = 1

batch_size = 60000 #for full batch
save_grad_norms= True

lr_init = 1e-1
optimizer_type = 'SGD'
lrscheduler_type = 'StepLR'
lrscheduler_args = {'step_size': 400,
                    'gamma': 0.1}


# for classical
epochs_classical = sum(epochs)
lr_init_classical = lr_init
lrscheduler_args_classical = {'step_size': 400,
                              'gamma': 0.1}
hidden_layers_classical = hidden_layers_start#no_iters + hidden_layers_start
fix_width_classical = 70 


# get data
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
no_steps_per_epoch = int(len(training_data)/batch_size)
print('no of iterations in one epoch:', no_steps_per_epoch)

end_list = []
for i, e in enumerate(epochs):
    end_list.append(int(e*len(training_data)/batch_size))
    end_list.append(1)
end_list.pop()  # removes last 1 which was too much

max_length = max(epochs_classical, sum(epochs))*no_steps_per_epoch

# create dataloader
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=10000)

# build models:

kwargs_net = {
    'hidden_layers': hidden_layers_start,
    'dim_hidden_layers': fix_width,
    'act_fun': act_fun,
    'type': _type
}

dim_in = 28*28
dim_out = 10

# kwargs_net_new = {  # classical net
#     'hidden_layers': hidden_layers_classical,
#     'dim_hidden_layers': fix_width_classical,
#     'act_fun': act_fun,  # nn.ReLU,  # TanhLU_shifted
#     'type': _typeexperiment_py_files/exp2.py
# }

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
    'dim_hidden_layers': fix_width_classical,
    'act_fun': act_fun,
    'type': _type
}


# determine which trainings are run
T1 = True
T2 = True
T3 = True
T4 = True
T5 = True
T6 = True
T7 = True
T8 = True
T9 = True
T10 = True


# define no of training run instances

no_of_initializations = 1  # 50

# set up empty lists for saving the observed quantities
# (besides the save to the json file)

final_testerror1 = []
final_testerror2 = []
final_testerror3 = []
final_testerror4 = []

# declare path where json files are saved

path1 = f'results_data/Exp{k}_1.json'
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
    param_init3 = copy.deepcopy(param_init.data)
    param_init4 = copy.deepcopy(param_init.data)
    param_init5 = copy.deepcopy(param_init.data)
    param_init6 = copy.deepcopy(param_init.data)
    param_init7 = copy.deepcopy(param_init.data)
    param_init8 = copy.deepcopy(param_init.data)
    param_init_class = copy.deepcopy(param_init.data)


    # train ali 1
    print('training on first ali')
    if T1:
        model1, mb_losses1, test_errors_short1, test_errors1, exit_flag1, grad_norm1 = layer_insertion_loop(
            iters=no_iters,
            epochs=epochs,
            model=model_init,
            kwargs_net=kwargs_net,
            dim_in=dim_in,
            dim_out=dim_out,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
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
        
        print(grad_norm1)
        # save losses1 and final accuracy1
        write_losses(path1,
                     mb_losses1, max_length, end_list, test_errors1,
                     interval_testerror=interval_testerror, exit_flag=exit_flag1,grad_norms = grad_norm1, its_per_epoch=no_steps_per_epoch)    # save losses3

        

    # build net for ali 2 and initalize with the parameters from ali 1
    if _type == 'fwd':
        model_init2 = feed_forward(dim_in, dim_out, **kwargs_net)
    if _type == 'res2':
        model_init2 = two_weight_resnet(dim_in, dim_out, **kwargs_net)
    if _type == 'res1':
        model_init2 = one_weight_resnet(dim_in, dim_out, **kwargs_net)

    torch.nn.utils.vector_to_parameters(param_init2, model_init2.parameters())

    # train ali 2
    print('training on second ali')
    if T2:
        model2, mb_losses2, test_errors_short2, test_errors2, exit_flag2, grad_norm2 = layer_insertion_loop(
            iters=no_iters,
            epochs=[20,50+60],
            model=model_init2,
            kwargs_net=kwargs_net,
            dim_in=dim_in,
            dim_out=dim_out,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
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

        # save losses2
        write_losses(f'results_data/Exp{k}_2.json',
                     mb_losses2, max_length, end_list, test_errors2, interval_testerror=interval_testerror, exit_flag=exit_flag2, grad_norms = grad_norm2,
                     its_per_epoch=no_steps_per_epoch)
        # full_list_of_losses_2.append(mb_losses2)
        final_testerror2.append(test_errors_short2[-1])


    # t3 ##########
    # build net for ali 2 and initalize with the parameters from ali 1
    if _type == 'fwd':
        model_init2 = feed_forward(dim_in, dim_out, **kwargs_net)
    if _type == 'res2':
        model_init2 = two_weight_resnet(dim_in, dim_out, **kwargs_net)
    if _type == 'res1':
        model_init2 = one_weight_resnet(dim_in, dim_out, **kwargs_net)

    torch.nn.utils.vector_to_parameters(param_init3, model_init2.parameters())

    # train ali 2
    print('training on second ali')
    if T3:
        model2, mb_losses2, test_errors_short2, test_errors2, exit_flag2, grad_norm2 = layer_insertion_loop(
            iters=no_iters,
            epochs=[30,50+50],
            model=model_init2,
            kwargs_net=kwargs_net,
            dim_in=dim_in,
            dim_out=dim_out,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
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

        # save losses2
        write_losses(f'results_data/Exp{k}_3.json',
                     mb_losses2, max_length, end_list, test_errors2, interval_testerror=interval_testerror, exit_flag=exit_flag2, grad_norms = grad_norm2,
                     its_per_epoch=no_steps_per_epoch)
        # full_list_of_losses_2.append(mb_losses2)
        final_testerror2.append(test_errors_short2[-1])


    # t4 ####################
    # build net for ali 2 and initalize with the parameters from ali 1
    if _type == 'fwd':
        model_init2 = feed_forward(dim_in, dim_out, **kwargs_net)
    if _type == 'res2':
        model_init2 = two_weight_resnet(dim_in, dim_out, **kwargs_net)
    if _type == 'res1':
        model_init2 = one_weight_resnet(dim_in, dim_out, **kwargs_net)

    torch.nn.utils.vector_to_parameters(param_init4, model_init2.parameters())

    # train ali 2
    print('training on second ali')
    if T4:
        model2, mb_losses2, test_errors_short2, test_errors2, exit_flag2, grad_norm2 = layer_insertion_loop(
            iters=no_iters,
            epochs=[40,50+40],
            model=model_init2,
            kwargs_net=kwargs_net,
            dim_in=dim_in,
            dim_out=dim_out,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
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

        # save losses2
        write_losses(f'results_data/Exp{k}_4.json',
                     mb_losses2, max_length, end_list, test_errors2, interval_testerror=interval_testerror, exit_flag=exit_flag2, grad_norms = grad_norm2,
                     its_per_epoch=no_steps_per_epoch)
        # full_list_of_losses_2.append(mb_losses2)
        final_testerror2.append(test_errors_short2[-1])

    # t5 #####################
    # build net for ali 2 and initalize with the parameters from ali 1
    if _type == 'fwd':
        model_init2 = feed_forward(dim_in, dim_out, **kwargs_net)
    if _type == 'res2':
        model_init2 = two_weight_resnet(dim_in, dim_out, **kwargs_net)
    if _type == 'res1':
        model_init2 = one_weight_resnet(dim_in, dim_out, **kwargs_net)

    torch.nn.utils.vector_to_parameters(param_init5, model_init2.parameters())

    # train ali 2
    print('training on second ali')
    if T5:
        model2, mb_losses2, test_errors_short2, test_errors2, exit_flag2, grad_norm2 = layer_insertion_loop(
            iters=no_iters,
            epochs=[50,50+30],
            model=model_init2,
            kwargs_net=kwargs_net,
            dim_in=dim_in,
            dim_out=dim_out,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
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

        # save losses2
        write_losses(f'results_data/Exp{k}_5.json',
                     mb_losses2, max_length, end_list, test_errors2, interval_testerror=interval_testerror, exit_flag=exit_flag2, grad_norms = grad_norm2,
                     its_per_epoch=no_steps_per_epoch)
        # full_list_of_losses_2.append(mb_losses2)
        final_testerror2.append(test_errors_short2[-1])


    # t6 # build net for ali 2 and initalize with the parameters from ali 1
    if _type == 'fwd':
        model_init2 = feed_forward(dim_in, dim_out, **kwargs_net)
    if _type == 'res2':
        model_init2 = two_weight_resnet(dim_in, dim_out, **kwargs_net)
    if _type == 'res1':
        model_init2 = one_weight_resnet(dim_in, dim_out, **kwargs_net)

    torch.nn.utils.vector_to_parameters(param_init6, model_init2.parameters())

    # train ali 2
    print('training on second ali')
    if T6:
        model2, mb_losses2, test_errors_short2, test_errors2, exit_flag2, grad_norm2 = layer_insertion_loop(
            iters=no_iters,
            epochs=[60,50+20],
            model=model_init2,
            kwargs_net=kwargs_net,
            dim_in=dim_in,
            dim_out=dim_out,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
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

        # save losses2
        write_losses(f'results_data/Exp{k}_6.json',
                     mb_losses2, max_length, end_list, test_errors2, interval_testerror=interval_testerror, exit_flag=exit_flag2, grad_norms = grad_norm2,
                     its_per_epoch=no_steps_per_epoch)
        # full_list_of_losses_2.append(mb_losses2)
        final_testerror2.append(test_errors_short2[-1])

    # t7 ############
    # build net for ali 2 and initalize with the parameters from ali 1
    if _type == 'fwd':
        model_init2 = feed_forward(dim_in, dim_out, **kwargs_net)
    if _type == 'res2':
        model_init2 = two_weight_resnet(dim_in, dim_out, **kwargs_net)
    if _type == 'res1':
        model_init2 = one_weight_resnet(dim_in, dim_out, **kwargs_net)

    torch.nn.utils.vector_to_parameters(param_init7, model_init2.parameters())

    # train ali 2
    print('training on second ali')
    if T7:
        model2, mb_losses2, test_errors_short2, test_errors2, exit_flag2, grad_norm2 = layer_insertion_loop(
            iters=no_iters,
            epochs=[70,50+10],
            model=model_init2,
            kwargs_net=kwargs_net,
            dim_in=dim_in,
            dim_out=dim_out,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
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

        # save losses2
        write_losses(f'results_data/Exp{k}_7.json',
                     mb_losses2, max_length, end_list, test_errors2, interval_testerror=interval_testerror, exit_flag=exit_flag2, grad_norms = grad_norm2,
                     its_per_epoch=no_steps_per_epoch)
        # full_list_of_losses_2.append(mb_losses2)
        final_testerror2.append(test_errors_short2[-1])

    # t8 #######################
    # build net for ali 2 and initalize with the parameters from ali 1
    if _type == 'fwd':
        model_init2 = feed_forward(dim_in, dim_out, **kwargs_net)
    if _type == 'res2':
        model_init2 = two_weight_resnet(dim_in, dim_out, **kwargs_net)
    if _type == 'res1':
        model_init2 = one_weight_resnet(dim_in, dim_out, **kwargs_net)

    torch.nn.utils.vector_to_parameters(param_init8, model_init2.parameters())

    # train ali 2
    print('training on second ali')
    if T8:
        model2, mb_losses2, test_errors_short2, test_errors2, exit_flag2, grad_norm2 = layer_insertion_loop(
            iters=no_iters,
            epochs=[80,50],
            model=model_init2,
            kwargs_net=kwargs_net,
            dim_in=dim_in,
            dim_out=dim_out,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
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

        # save losses2
        write_losses(f'results_data/Exp{k}_8.json',
                     mb_losses2, max_length, end_list, test_errors2, interval_testerror=interval_testerror, exit_flag=exit_flag2, grad_norms = grad_norm2,
                     its_per_epoch=no_steps_per_epoch)
        # full_list_of_losses_2.append(mb_losses2)
        final_testerror2.append(test_errors_short2[-1])





    # t9 #####################

    # build net for classical small
    if _type == 'fwd':
        model_classical = feed_forward(dim_in, dim_out, **kwargs_net_classical)
    if _type == 'res2':
        model_classical = two_weight_resnet(
            dim_in, dim_out, **kwargs_net_classical)
    if _type == 'res1':
        model_classical = one_weight_resnet(
            dim_in, dim_out, **kwargs_net_classical)
        
    torch.nn.utils.vector_to_parameters(param_init_class, model_classical.parameters())

    # build optimizer and lr scheduler for the classical training:
    # build optimizer
    if optimizer_type == 'SGD':
        optimizer_classical = torch.optim.SGD(
            model_classical.parameters(), lr_init_classical)

    # build lr scheduler
    if lrscheduler_type == 'StepLR':
        step_size = lrscheduler_args_classical['step_size']
        gamma = lrscheduler_args_classical['gamma']
        lrscheduler_classical = torch.optim.lr_scheduler.StepLR(
            optimizer_classical, step_size=step_size, gamma=gamma)





    # train classical  small
    print('classical training small!')
    if T9:
        print('training classically on model', model_classical)
        mblosses_classical, lr_end, test_error_classical, exit_flag3, grad_norm3 = train(model_classical,
                                                                 train_dataloader=train_dataloader,
                                                                 epochs=epochs_classical,
                                                                 optimizer=optimizer_classical,
                                                                 scheduler=lrscheduler_classical,
                                                                 wanted_testerror=wanted_testerror,
                                                                 start_with_backtracking=None,
                                                                 check_testerror_between=interval_testerror,
                                                                 test_dataloader=test_dataloader,
                                                                 print_param_flag=False,
                                                                 save_grad_norms=save_grad_norms
                                                                 )

        # save losses3
        write_losses(f'results_data/Exp{k}_9.json', mblosses_classical, max_length, structures=[
                    epochs_classical], errors=test_error_classical,
                    interval_testerror=interval_testerror, exit_flag=exit_flag3, grad_norms = grad_norm3,
                    its_per_epoch=no_steps_per_epoch)

        
    # next step in loop


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

    # train classical  small
    print('classical training big!')
    if T10:
        print('training classically on model', model_classical2)
        mblosses_classical2, lr_end2, test_error_classical2, exit_flag4, grad_norm4 = train(model_classical2,
                                                                 train_dataloader=train_dataloader,
                                                                 epochs=epochs_classical,
                                                                 optimizer=optimizer_classical2,
                                                                 scheduler=lrscheduler_classical2,
                                                                 wanted_testerror=wanted_testerror,
                                                                 start_with_backtracking=None,
                                                                 check_testerror_between=interval_testerror,
                                                                 test_dataloader=test_dataloader,
                                                                 print_param_flag=False,
                                                                 save_grad_norms=save_grad_norms
                                                                 )

        # save losses3
        write_losses(f'results_data/Exp{k}_10.json', mblosses_classical2, max_length, structures=[
                    epochs_classical], errors=test_error_classical2,
                    interval_testerror=interval_testerror, exit_flag=exit_flag4,
                    grad_norms= grad_norm4,
                    its_per_epoch=no_steps_per_epoch)

        
