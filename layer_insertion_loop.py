from typing import Dict
from random import seed, sample
import torch
import spirals
import slucks
import stalling
from train_and_test import check_testerror, train_model_on_spiral_freezed_params
from model_selection import tmp_net, select_new_model
from calc_shadow_prices import calculate_shadowprices_for_one_gradient
from model_utils import freeze_old_params


def train_adaptive_freezing(
        model, kwargs_net, kwargs_spiral: Dict, loss_fn, bs, lr, epochs=50000, callback=None,
        iters=5, wanted_test_error=0.5, use_adaptive_lr=False, mode='abs max', save_decbound_after_li=False,
        stall_detector=stalling.stall_detector(300, 1000, 0, 500), check_testerror_between=None):
    '''
    implements training loop for (adaptive) layer insertion and minibatch SGD

    Args:
            model: pytorch feedfwd net
            kwargs_net (dict): keywordargs for building the model (feedforward net)
            kwargs_spiral (dict): keywordargs for building the dataset (2d,2class spiral)
            loss_fn: loss function
            bs (int): batchsize
            lr (float): learning rate (fixed)
            epochs (int): maximum number of epochs which are performed during the classical learning
            stages of the loop. default 5
            callback: ? for plotting intermediate dec boundaries
            iters (int): number of layer instertion loops which are performed at most. default 5
            wanted_test_error (float): test error until we want to train (optimally). default 0.5
            use_adaptive_lr (bool): dtermines whether fixed or adaptive lr is used
            mode (string): either 'min' or 'max' so far, indicating whether the layer with th largest (theory)
            or smallest (comparison) lagrange multipliers are chosen in model selection. default is 'max'.

    Out:
        model: trained model (of a a-priori unknown architecture)
        mb_losses_total (list): list containing the minibatch losses during the adaptive training loop
        test_err_list (list): list of test errors after each "step 1"-training
        end_of_training_steps (list): list which holds information how many epochs were performed in each loop

    '''
    print_param_flag = False
    if not kwargs_spiral:
        kwargs_spiral = {
            'number': 100,
            'circles': 3,
            'r0': .1
        }

    if type(epochs) == int:
        epochs = (iters+1)*[epochs]
    if len(epochs) < iters+1:  # if not enough epoch numbers are given, they will be filled with the last entry
        diff = iters+1-len(epochs)
        epochs = epochs + diff * [epochs[-1]]

    # if not kwargs_net:
    #    kwargs_net = {
    #        'hidden_layers': 2,
    #        'dim_hidden_layers': 50
    #    }

    # model = net.feed_forward(**kwargs_net)#.to(device) # build initial model
    reds, blues = spirals.gen_spiral(**kwargs_spiral)  # build spiral
    reds = reds  # .to(device)
    blues = blues  # .to(device)

    data_x = torch.cat((reds, blues))  # build training input data
    # build training output data
    data_y = torch.tensor([0] * len(reds) + [1] * len(blues))
    dataset_size = len(data_x)

    no_of_test_points = int(0.25*dataset_size)
    seed(1)  # set seed in splitting of test and training data for reproducibility

    all_indices = range(0, dataset_size)
    # generate test batch indices
    test_indices = sample(all_indices, no_of_test_points)
    train_indices = list(all_indices)
    for e in test_indices:
        train_indices.remove(e)

    test_data_x = data_x[test_indices]  # generate test data
    test_data_y = data_y[test_indices]

    train_data_x = data_x[train_indices]  # generate training data
    train_data_y = data_y[train_indices]

    mb_losses_total = []
    test_err_list = []
    test_err_list2 = []
    end_of_training_steps = []

    for k in range(iters):
        # iterate on current net
        print(f'starting on {k+1}. net !')
        print(model)
        if k == 1:
            print_param_flag = True
        norms1, freezed_norms1, mb_losses1, lr, test_error_l2 = train_model_on_spiral_freezed_params(
            model, loss_fn, training_data_in=train_data_x, training_data_out=train_data_y,
            stall_detector=stall_detector, epochs=epochs[k], bs=bs, lr=lr, use_adaptive_lr=use_adaptive_lr,
            freezed=[], start_with_backtracking=True, save_decbound_after_li=save_decbound_after_li,
            check_testerror_between=check_testerror_between, test_data_x=data_x, test_data_y=data_y,
            wanted_testerror=wanted_test_error, print_param_flag=print_param_flag)
        # train classically until stalling
        mb_losses_total = mb_losses_total + mb_losses1
        end_of_training_steps.append(len(mb_losses1))
        test_err_list2 = test_err_list2+test_error_l2
        # test error check if good, break, if not, write in list
        curr_test_err = check_testerror(test_data_x, test_data_y, model)
        test_err_list.append(curr_test_err)
        test_err_list2.append(curr_test_err)
        print(f'Test error after first step of loop {k+1} is {curr_test_err}!')
        if curr_test_err <= wanted_test_error:
            print(f'The final model has the architecture: {model}')
            if False:
                slucks.plot_decision_boundary(
                    model, train_data_x, train_data_y)
            return model, mb_losses_total, test_err_list, end_of_training_steps, test_err_list2
        # if callback:
        #    callback(model, reds, blues) #?
        if False:
            slucks.plot_decision_boundary(model, train_data_x, train_data_y)
        # slucks.plot_decision_boundary(model,test_data_x,test_data_y)

        # build partially frozen net for the shadow prices
        # build temporary net for the equality constrained training in next step
        model, freezed, kwargs_net = tmp_net(
            hidden_layers=kwargs_net['hidden_layers'],
            dim_hidden_layers=kwargs_net['dim_hidden_layers'],
            act_fun=kwargs_net['act_fun'],
            _type=kwargs_net['type'],
            model=model,
            training_data=train_data_x)

        # train one epoch to determine the shadow prices
        # norms, freezed_norms, mb_losses2 = train_model_on_spiral_freezed_params(model, loss_fn,
        # training_data_in=train_data_x, training_data_out=train_data_y, stall_detector=stall_detector,
        # epochs = 1, bs = bs, lr = lr,use_adaptive_lr=use_adaptive_lr, freezed=freezed)
        # # 1 epoch equality constrained training

        # alternative: only one full batch gradient
        norms, freezed_norms, mb_losses2 = calculate_shadowprices_for_one_gradient(
            model, loss_fn, train_data_x, train_data_y, freezed)

        # for plotting/observation purposes
        # print('norms: ',norms)
        # print(' freezed norms: ',freezed_norms)
        mb_losses_total = mb_losses_total+mb_losses2
        end_of_training_steps.append(len(mb_losses2))

        # select new model based on the shadow prices
        # insert one frozen layer and unfreeze, delete other frozen layers

        # works only for resnets so far!!!!! TODO
        model, kwargs_net, new_child = select_new_model(
            norms, freezed_norms, model=model, freezed=freezed, kwargs_net=kwargs_net, mode=mode,
            _type=kwargs_net['type'])

        # train only new parameters
        train_only_new = False  # works only for resnets so far!! TODO
        if train_only_new:
            # old_parameters =0 liste ohne neues layer
            old_parameters = freeze_old_params(model, new_child)
            epochs2 = int(0.1*epochs[k])
            stalling2 = stall_detector
            a, b, c, d, e = train_model_on_spiral_freezed_params(model, loss_fn, train_data_x, train_data_y, stalling2,
                                                                 epochs2,
                                                                 bs=300, lr=lr, use_adaptive_lr=use_adaptive_lr,
                                                                 freezed=old_parameters, start_with_backtracking=True,
                                                                 save_decbound_after_li=save_decbound_after_li,
                                                                 check_testerror_between=check_testerror_between,
                                                                 test_data_x=data_x, test_data_y=data_y,
                                                                 wanted_testerror=wanted_test_error,
                                                                 print_param_flag=print_param_flag)
            mb_losses_total = mb_losses_total + c
            end_of_training_steps.append(len(c))
            test_err_list2 = test_err_list2+e
        # lr = backtracking.simple_backtracking(model,lr,0.9,mb_losses2[-1],loss_fn,train_data_x,train_data_y,max_it=20)
        lr *= .8  # decrease lr for next loop

    print(f'starting on {k+2}. net!')
    print(model)
    normslast, freezednormslast, mb_losseslast, lr, test_error_l2 = train_model_on_spiral_freezed_params(
        model, loss_fn, training_data_in=train_data_x, training_data_out=train_data_y, stall_detector=stall_detector,
        epochs=epochs[k], bs=bs, lr=lr, use_adaptive_lr=use_adaptive_lr, freezed=[
        ], start_with_backtracking=True,
        save_decbound_after_li=save_decbound_after_li, check_testerror_between=check_testerror_between,
        test_data_x=test_data_x, test_data_y=test_data_y,
        wanted_testerror=wanted_test_error, print_param_flag=print_param_flag)  # after the last li train again

    mb_losses_total = mb_losses_total + mb_losseslast
    end_of_training_steps.append(len(mb_losseslast))
    test_err_list2 = test_err_list2 + test_error_l2
    curr_test_err = check_testerror(test_data_x, test_data_y, model)
    test_err_list.append(curr_test_err)
    test_err_list2.append(curr_test_err)
    print(f'Test error of loop {k+2} is {curr_test_err}!')
    # if callback:
    #        callback(model, reds, blues)
    if False:
        slucks.plot_decision_boundary(model, train_data_x, train_data_y)
    print(f'The final model has the architecture: {model}')
    stall_detector.reset()
    return model, mb_losses_total, test_err_list, end_of_training_steps, test_err_list2
