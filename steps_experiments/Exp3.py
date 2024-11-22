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
from stepfunction_dataset import gen_steps_dataset, plot_predictor_withdataset

# ################# fix hyperparameters ###################################

# for checking the progress of the training in the terminal, use the bash command: jp length filename.json
# to see how many runs are already saved

k = 3

# seed
s=1
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)

torch.set_num_threads(8)

plot = False

# Define hyperparameters

hidden_layers_start = 2
fix_width = [3]#[5]
no_iters = 5
lr_decrease_after_li = 1.
epochs = [500,500,500,500,500,500]#,5000,5000,5000,5000]  # [10, 5, 5]
wanted_testerror = 1e-4
_type = 'fwd'
act_fun = nn.ReLU
interval_testerror = 1

batchsize =375
no_points = 500
n_steps = 5


td, vd, data_X, data_y = gen_steps_dataset(n_points=no_points,n_steps= n_steps,noise=0.,seed =0,batchsize=batchsize)

loss_fn = torch.nn.MSELoss()
test_mode = 'mse'

no_steps_per_epoch = 8#50
print('no of iterations in one epoch:', no_steps_per_epoch)

end_list = []
for i, e in enumerate(epochs):
    end_list.append(e)
    end_list.append(1)
end_list.pop()  # removes last 1 which was too much




save_grad_norms= True
save_heatmaps = True

lr_init = 1e-2
optimizer_type = 'SGD'
lrscheduler_type = 'StepLR'
lrscheduler_args = {'step_size': 40000,
                    'gamma': 0.1}


# for classical
epochs_classical = sum(epochs)
lr_init_classical = lr_init
lrscheduler_args_classical = {'step_size': 40000,
                              'gamma': 0.1}
hidden_layers_classical = hidden_layers_start#no_iters + hidden_layers_start
fix_width_classical = fix_width

# build models:

kwargs_net = {
    'hidden_layers': hidden_layers_start,
    'dim_hidden_layers': fix_width,
    'act_fun': act_fun,
    'type': _type,
    'flatten': False
}

dim_in = 1
dim_out = 1


# classical net small
kwargs_net_classical = {
    'hidden_layers': hidden_layers_classical,
    'dim_hidden_layers': fix_width_classical,
    'act_fun': act_fun,
    'type': _type,
    'flatten': False
}

# classical net small
kwargs_net_classical2 = {
    'hidden_layers': hidden_layers_classical+1,
    'dim_hidden_layers': fix_width_classical+[fix_width_classical[-1]],
    'act_fun': act_fun,
    'type': _type,
    'flatten': False
}


max_length = max(epochs_classical, sum(epochs))*no_steps_per_epoch

# determine which trainings are run
T1 = True
T2 = True
T3 = True
T4 = False

# define no of training run instances

no_of_initializations = 1  # 50

# set up empty lists for saving the observed quantities
# (besides the save to the json file)

final_testerror1 = []
final_testerror2 = []
final_testerror3 = []
final_testerror4 = []

# declare path where json files are saved

path1 = f'results_data_steps/Exp{k}_1.json'
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
    if plot:
        plot_predictor_withdataset(data_X,data_y,model_init, device='cpu',title='model_init',save=False,only_data=False)
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
            loss_fn=loss_fn,
            test_mode = test_mode,
            save_heatmaps=save_heatmaps
        )
        final_params = torch.nn.utils.parameters_to_vector(model1.parameters()).detach().numpy().tolist()
        if plot:
            plot_predictor_withdataset(data_X,data_y,model1, device='cpu',title='model1',save=False,only_data=False)
            #print(grad_norm1)
            plt.plot(mb_losses1)
            plt.yscale('log')
            plt.show()
            plt.plot(test_errors1)
            plt.yscale('log')
            plt.show()

        # save losses1 and final accuracy1
        write_losses(path1,
                    mb_losses1, max_length, end_list, test_errors1,
                    interval_testerror=interval_testerror, times=times1,grad_norms = grad_norm1, its_per_epoch=no_steps_per_epoch,
                    final_params=final_params, sens=sens1)    # save losses3
        

        # full_list_of_losses_1.append(mb_losses)
        final_testerror1.append(test_errors_short1[-1])

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
        if plot:
            plot_predictor_withdataset(data_X,data_y,model_init2, device='cpu',title='model_init2',save=False,only_data=False)
        model2, mb_losses2, test_errors_short2, test_errors2, exit_flag2, grad_norm2, times2, sens2 = layer_insertion_loop(
            iters=no_iters,
            epochs=epochs,
            model=model_init2,
            kwargs_net=kwargs_net,
            dim_in=dim_in,
            dim_out=dim_out,
            train_dataloader=td,
            test_dataloader=vd,
            lr_init=lr_init,
            wanted_test_error=wanted_testerror,
            mode='abs min',
            optimizer_type=optimizer_type,
            lrschedule_type=lrscheduler_type,
            lrscheduler_args=lrscheduler_args,
            check_testerror_between=interval_testerror,
            decrease_after_li=lr_decrease_after_li,
            print_param_flag=False,
            start_with_backtracking=None,
            v2=False,
            save_grad_norms=save_grad_norms,
            loss_fn=loss_fn,
            test_mode = test_mode,
            save_heatmaps=save_heatmaps
        )
        final_params = torch.nn.utils.parameters_to_vector(model2.parameters()).detach().numpy().tolist()
        if plot:
            plot_predictor_withdataset(data_X,data_y,model2, device='cpu',title='model2',save=False,only_data=False)
            plt.plot(mb_losses2)
            plt.yscale('log')
            plt.show()
            plt.plot(test_errors2)
            plt.yscale('log')
            plt.show()

        # save losses2
        write_losses(f'results_data_steps/Exp{k}_2.json',
                    mb_losses2, max_length, end_list, test_errors2, interval_testerror=interval_testerror, times=times2, grad_norms = grad_norm2,
                    its_per_epoch=no_steps_per_epoch, final_params=final_params, sens=sens2)
        # full_list_of_losses_2.append(mb_losses2)
        final_testerror2.append(test_errors_short2[-1])

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
    if T3:
        if plot:
            plot_predictor_withdataset(data_X,data_y,model_classical, device='cpu',title='model_classical',save=False,only_data=False)
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
                                                                 save_grad_norms=save_grad_norms,
                                                                 loss_fn=loss_fn,
                                                                 test_mode = test_mode,
                                                                 save_heatmaps=save_heatmaps
                                                                 )
        
        final_params = torch.nn.utils.parameters_to_vector(model_classical.parameters()).detach().numpy().tolist()
        if plot: 
            plot_predictor_withdataset(data_X,data_y,model_classical, device='cpu',title='model_classical',save=False, only_data=False)
            plt.plot(mblosses_classical)
            plt.yscale('log')
            plt.show()
            plt.plot(test_error_classical)
            plt.yscale('log')
            plt.show()
        # save losses3
        write_losses(f'results_data_steps/Exp{k}_3.json', mblosses_classical, max_length, structures=[
                   epochs_classical], errors=test_error_classical,
                   interval_testerror=interval_testerror, times=times3, grad_norms = grad_norm3,
                   its_per_epoch=no_steps_per_epoch, final_params=final_params)
        
        # full_list_of_losses_3.append(mb_losses_comp)
        final_testerror3.append(check_testerror(
            vd, model_classical, test_mode=test_mode))
        print(f'test error of classical training small {final_testerror3[-1]}')
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
    
    if T4:
        print('classical training big!')
        if plot: 
            plot_predictor_withdataset(data_X,data_y,model_classical2, device='cpu',title='model_classical2',save=False, only_data=False)
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
                                                                 save_grad_norms=save_grad_norms,
                                                                 loss_fn=loss_fn,
                                                                 test_mode = test_mode,
                                                                 save_heatmaps=save_heatmaps
                                                                 )
        
        final_params = torch.nn.utils.parameters_to_vector(model_classical2.parameters()).detach().numpy().tolist()
        if plot:
            plot_predictor_withdataset(data_X,data_y,model_classical2, device='cpu',title='model_classical2',save=False, only_data=False)
            plt.plot(mblosses_classical2)
            plt.yscale('log')
            plt.show()
            plt.plot(test_error_classical2)
            plt.yscale('log')
            plt.show()

        # save losses3
        write_losses(f'results_data_steps/Exp{k}_4.json', mblosses_classical2, max_length, structures=[
                   epochs_classical], errors=test_error_classical2,
                   interval_testerror=interval_testerror, times=times4,
                   grad_norms= grad_norm4,
                   its_per_epoch=no_steps_per_epoch, final_params=final_params)

        # full_list_of_losses_3.append(mb_losses_comp)
        final_testerror4.append(check_testerror(
            vd, model_classical2))
        print(f'test error of classical training big {final_testerror4[-1]}')
    # next step in loop

plt.plot(mb_losses1, label='ali1')
plt.plot(mb_losses2, label='ali2')
plt.plot(mblosses_classical, label='classical1')
plt.yscale('log')
plt.legend()
plt.show()

plt.plot(test_errors1, label='ali1')
plt.plot(test_errors2, label='ali2')
plt.plot(test_error_classical, label='classical1')
plt.yscale('log')
plt.legend()
plt.show()