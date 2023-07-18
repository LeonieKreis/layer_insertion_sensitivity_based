#  ########## necessary imports ###############################################
from random import seed, sample
import os.path
from matplotlib import pyplot as plt
# from activation_functions import TanhLU, TanhLU_shifted
import stalling
import spirals
import net
import adaptive_layer_insertion
import train_and_test
import torch
from torch import nn
import json
from plot_helper import getx_coords_of_error
torch.set_num_threads(8)

# from gif_maker import make_gif
# from utils import ema

# ################# fix hyperparameters ###################################

# for checking the progress of the training in the terminal, use the bash command: jp length filename.json
# to see how many runs are already saved

k = 1

seed(1)
# TODO set torch seed fixed as well

def write_losses(path, losses, max_length, structures=None, errors=None, number=None, interval_testerror=None):
    '''
    saves losses in json file in a dict,
    optionally saves also the information which loss happened on which model
    when structures is given. In order to have the same length, for the std and mean function,
    nan values get appended to the end of losses until the list has the length max_length.
    Then we use mean and std methods, which filter out nan values for their compuations
    '''

    try:
        with open(path) as file:
            data = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        data = {}

    if len(losses) < max_length:
        diff = max_length - len(losses)
        losses += diff*[float("nan")]

    # save errors at the right points
    errs = (max_length+1)*[float("nan")]
    ind = getx_coords_of_error(structures, interval_testerror)
    for i, e in zip(ind, errors):
        errs[i] = e

    if number is None:
        number = len(data.keys())
    if structures is None and errors is None:
        data[str(number)] = {'losses': losses}
    if structures is not None and errors is None:
        data[str(number)] = {'losses': losses,
                             'structures': structures}
    if structures is None and errors is not None:
        data[str(number)] = {'losses': losses,
                             'errors': errs}
    if structures is not None and errors is not None:
        data[str(number)] = {'losses': losses,
                             'structures': structures,
                             'errors': errs}

    with open(path, 'w') as file:
        json.dump(data, file)


hidden_layers_start = 1
fix_width = 10
no_iters = 3
epochs_per_loop = 50000
wanted_test_error = 0.
interval_between = 100

hidden_layers_classical = no_iters + hidden_layers_start

# get data
# TODO
# split into test and training data (same for all trainings)

kwargs_net = {
    'hidden_layers': hidden_layers_start,
    'dim_hidden_layers': fix_width,
    'act_fun': nn.Tanh,  # nn.ReLU,  # TanhLU_shifted #nn.ReLU
    'type': 'res2'
}

kwargs_net_new = {  # classical net
    'hidden_layers': hidden_layers_classical,
    'dim_hidden_layers': fix_width,
    'act_fun': nn.Tanh,  # nn.ReLU,  # TanhLU_shifted
    'type': 'res2'
}

lr_init = 1e-2  # 0.02#0.05
bs = 300  # *40*0.75

lr_for_classical = lr_init
epochs_classical = (no_iters+1)*epochs_per_loop + no_iters
max_length = epochs_classical


# ################ perform training for many initializations absmax, absmin, 3layernet ##############################

no_of_initializations = 50

full_list_of_losses_1 = []
full_list_of_losses_2 = []
full_list_of_losses_3 = []

full_list_of_ends_1 = []
full_list_of_ends_2 = []
full_list_of_ends_3 = []

final_testerror1 = []
final_testerror2 = []
final_testerror3 = []

path1 = f'../results_data/Exp{k}_1.json'
if os.path.isfile(path1):
    print(f' file with path {path1} already exists!')
    quit()


for i in range(no_of_initializations):
    print(f'loop number {i}!')
    seed()  # random initilaization for each model starting parameters
    # build net for ali 1
    model_init = 0 # TODO
    param_init = torch.nn.utils.parameters_to_vector(model_init.parameters())

    # train ali 1
    print('training on first ali')
    #TODO

    # save losses1 and final accuracy1
    write_losses(path1,
                 mb_losses, max_length, end_list, error_list, interval_testerror=interval_between)
    # full_list_of_losses_1.append(mb_losses)
    final_testerror1.append(test_error_list[-1])

    # build net for ali 2 and initalize with the parameters from ali 1
    model_init2 = 0 # TODO
    torch.nn.utils.vector_to_parameters(param_init, model_init2.parameters())

    # train ali 2
    print('training on second ali')
    #TODO

    # save losses2
    write_losses(f'../results_data/Exp{k}_2.json',
                 mb_losses2, max_length, end_list2, error_list2, interval_testerror=interval_between)
    # full_list_of_losses_2.append(mb_losses2)
    final_testerror2.append(test_error_list2[-1])

    # build net for classical
    model_comp = net.feed_forward(**kwargs_net_new)

    # train classical
    print('classical training!')
    #TODO

    # save losses3
    write_losses(f'../results_data/Exp{k}_3.json',
                 mb_losses_comp, max_length, structures=[
                     epochs_classical], errors=error_between,
                 interval_testerror=interval_between)

    # full_list_of_losses_3.append(mb_losses_comp)
    final_testerror3.append(train_and_test.check_testerror(
        test_data_x, test_data_y, model_comp))
    print(f'test error of classical training {final_testerror3[-1]}')
    # next step in loop


