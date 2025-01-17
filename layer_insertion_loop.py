import torch
import sys
import time

from utils import get_timestamp
from train_and_test_ import check_testerror, train
from model_selection import tmp_net, select_new_model
from calculate_shadow_prices import calculate_shadowprices_minibatch


def layer_insertion_loop(
        iters, epochs, model, kwargs_net, dim_in, dim_out, train_dataloader, test_dataloader, lr_init,
        wanted_test_error=0.5, mode='abs max', optimizer_type='SGD', lrschedule_type='StepLR', lrscheduler_args=None,
        check_testerror_between=None, decrease_after_li=1., print_param_flag=False, start_with_backtracking=None,
        v2=False, save_grad_norms=False, use_adaptive_lr=False):
    '''
    implements training loop for (adaptive) layer insertion and minibatch SGD

    Args:
            iters (int): number of layer insertions which are done
            epochs (int or list): maximum number of epochs which are performed during the learning
            model: pytorch net. should be either a resnet with two weights or a feedforward net 
            kwargs_net (dict): dict specifying the 4 keys hidden_layers, dim_hidden_layers, act_fun and type
            dim_in (int): (flattened) dimesion of the images
            dim_out (int): number of classes
            train_dataloader: iterable from pytorch containing the training data
            test_dataloader: iterable from pytorch containing the test data
            lr_init : inital learning rate
            wanted_test_error (float): test error until we want to train (optimally). default 0.5
            mode (string): either 'abs min' or 'abs max' , 'pos 0' or 'threshold, indicating whether the layer with th largest (theory)
            or smallest (comparison) lagrange multipliers are chosen in model selection. default is 'abs max'
            optimizer_type (string): so far only 'SGD' is implemented
            lrschedule_type (string): so far only 'StepLR' is implemented
            lrscheduler_args (default None): dict which contains the hyperparameters needed by the lrscheduler
            check_testerror_between (None or int): if none, the testerror is noct checked while training on a model.
            If an integer is specified, e.g. k=1, the after k epochs, the testerror is checked during training on one model.
            decrease_after_li: factor by which the current lr is decreased after a new layer is inserted.
            print_param_flag (default False): if True, prints the parameter gradients for the first 10 epochs
            start_with_backtracking (None or int): if None, there is  no backtracking performed, if it is an integer k ,
            then for the first k epochs, backtracking is performed after the layer insertion.
            v2: default: False
            save_grad_norms: (bool) default False. If True, saves the averaged layerwise squared norm of the gradient in each step of the optimizer during training.
            use_adaptive_lr (bool): default False. If True, uses an adaptive learning rate scheme suggested by Frederik Köhne


    Out:
        model: trained model (of a a-priori unknown architecture)
        mb_losses_total (list): list containing the minibatch losses during the adaptive training loop
        test_err_list (list): list of test errors after each "step 1"-training on a model
        test_err_list2 (list): list of test errors computed also during the training
        exit_flag: 0 or 1. 1 indicates that wanted testerror was reached during training
        grad_norms_total: list of list ov layerwise avergae gradients for all iterations
        times_total: times needed for each iteration

    '''

    if type(epochs) == int:
        epochs = (iters+1)*[epochs]
    if len(epochs) < iters+1:  # if not enough epoch numbers are given, they will be filled with the last entry
        diff = iters+1-len(epochs)
        epochs = epochs + diff * [epochs[-1]]

    mb_losses_total = []
    times_total = []
    grad_norms_total = []
    test_err_list = []
    test_err_list2 = []
    lr = lr_init
    exit_flag = 0  # means that wanted testerror is not attained
    end_time = 0

    for k in range(iters):
        # iterate on current net
        print(f'starting on {k+1}. net !')
        print(model)
        # build optimizer
        if optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr)
        if optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=5e-3)

        # build lr scheduler
        if lrschedule_type == 'StepLR':
            if isinstance(lrscheduler_args['step_size'],list):
                step_size = lrscheduler_args['step_size'][k]
            else:
                step_size = lrscheduler_args['step_size']
            gamma = lrscheduler_args['gamma']
            lrscheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma)
            
        if lrschedule_type == 'MultiStepLR':
            if isinstance(lrscheduler_args['step_size'][0],list):
                step_size = lrscheduler_args['step_size'][k]
            else:
                step_size = lrscheduler_args['step_size']
            gamma = lrscheduler_args['gamma']
            lrscheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=step_size, gamma=gamma)

        # train
        mb_losses1, lr_end_lastloop, test_error_l2, exit_flag, grad_norms, times1 = train(
            model, train_dataloader, epochs[k], optimizer, lrscheduler, wanted_testerror=wanted_test_error,
            start_with_backtracking=start_with_backtracking, check_testerror_between=check_testerror_between,
            test_dataloader=test_dataloader, print_param_flag=print_param_flag, save_grad_norms=save_grad_norms, use_adaptive_lr=use_adaptive_lr)

        # train classically until stalling
        mb_losses_total = mb_losses_total + mb_losses1
        times1_updated = [i + end_time for i in times1]
        times_total = times_total + times1_updated
        # grad_norms_total = grad_norms_total + grad_norms
        grad_norms_total.append(grad_norms)

        

        test_err_list2 = test_err_list2+test_error_l2

        # test error check if good, break, if not, write in list
        curr_test_err = check_testerror(test_dataloader, model)
        test_err_list.append(curr_test_err)
        print(f'Test error after first step of loop {k+1} is {curr_test_err}!')
        if curr_test_err <= wanted_test_error:
            exit_flag = 1
            print(f'The final model has the architecture: {model}')
            return model, mb_losses_total, test_err_list, test_err_list2, exit_flag, grad_norms_total

        # get time of layer selection and new initialization
        tic = time.time()


        # build partially frozen net for the shadow prices
        # build temporary net for the equality constrained training in next step
        model_tmp, freezed, kwargs_net_tmp = tmp_net(
            dim_in=dim_in,
            dim_out=dim_out,
            hidden_layers=kwargs_net['hidden_layers'],
            dim_hidden_layers=kwargs_net['dim_hidden_layers'],
            act_fun=kwargs_net['act_fun'],
            _type=kwargs_net['type'],
            model=model,
            v2=v2)

        # compute sensitivities
        free_norms, freezed_norms, mb_losses2 = calculate_shadowprices_minibatch(
            train_dataloader, model_tmp, freezed)

        # mb_losses_total = mb_losses_total+mb_losses2
        # mb_losses_total = mb_losses_total + [sum(mb_losses2)/len(mb_losses2)] # uncommenting this gives a double iterate

        # select new model based on the shadow prices
        # insert one frozen layer and unfreeze, delete other frozen layers
        model, kwargs_net, new_child, sens = select_new_model(
            free_norms, freezed_norms, model=model_tmp, freezed=freezed, kwargs_net=kwargs_net_tmp, mode=mode,
            _type=kwargs_net['type'], v2=v2)
        
        toc = time.time()

        time_model_selection = toc-tic
        times_total[-1]= times_total[-1]+time_model_selection

        end_time = times_total[-1]

        if save_grad_norms:
            values_at_li = []
            for p in model.parameters():
                if p.requires_grad:
                    values_at_li.append(torch.linalg.norm(p.data))
                    print(values_at_li)

            original_stdout = sys.stdout
            path = f'val_at_li{get_timestamp()}.txt'
            with open(path, 'w') as f:
                sys.stdout = f
                print(
                    f'norm of values of parameters (parameter-wise) at layer insertion: {values_at_li}')
                # Reset the standard output
                sys.stdout = original_stdout

        lr = decrease_after_li * lr_end_lastloop  # decrease lr for next loop

    print(f'starting on {k+2}. net!')
    print(model)

    # build optimizer
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr)
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),lr, weight_decay=5e-3)

    # build lr scheduler
    if lrschedule_type == 'StepLR':
        step_size = lrscheduler_args['step_size']
        gamma = lrscheduler_args['gamma']
        lrscheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
        
    if lrschedule_type == 'MultiStepLR':
        if isinstance(lrscheduler_args['step_size'][0],list):
            step_size = lrscheduler_args['step_size'][k+1]
        else:
            step_size = lrscheduler_args['step_size']
        gamma = lrscheduler_args['gamma']
        lrscheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=step_size, gamma=gamma)

    mb_losseslast, lr, test_error_l2, exit_flag, grad_norms, times2 = train(model, train_dataloader, epochs[k+1], optimizer, lrscheduler,
                                                                    wanted_testerror=wanted_test_error,
                                                                    start_with_backtracking=start_with_backtracking,
                                                                    check_testerror_between=check_testerror_between,
                                                                    test_dataloader=test_dataloader, print_param_flag=print_param_flag,
                                                                    save_grad_norms=save_grad_norms,use_adaptive_lr=use_adaptive_lr)
    # after the last li train again

    mb_losses_total = mb_losses_total + mb_losseslast

    times2_updated = [i + end_time for i in times2]

    times_total = times_total + times2_updated
    # grad_norms_total = grad_norms_total + grad_norms
    grad_norms_total.append(grad_norms)

    test_err_list2 = test_err_list2 + test_error_l2
    curr_test_err = check_testerror(test_dataloader, model)
    if curr_test_err <= wanted_test_error:
        exit_flag = 1
    test_err_list.append(curr_test_err)

    print(f'Test error of loop {k+2} is {curr_test_err}!')
    print(f'The final model has the architecture: {model}')
    print(
        f'norm of values of parameters (parameter-wise) at layer insertion: {values_at_li}')

    return model, mb_losses_total, test_err_list, test_err_list2, exit_flag, grad_norms_total, times_total, sens
