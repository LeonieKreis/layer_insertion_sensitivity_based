import torch
from train_and_test_ import check_testerror, train
from model_selection import tmp_net, select_new_model
from calculate_shadow_prices import calculate_shadowprices_minibatch


def layer_insertion_loop(
        iters, epochs, model, kwargs_net, dim_in, dim_out, train_dataloader, test_dataloader, lr_init,
        wanted_test_error=0.5, mode='abs max', optimizer_type='SGD', lrschedule_type='StepLR', lrscheduler_args=None,
        check_testerror_between=None, decrease_after_li=1., print_param_flag=False, start_with_backtracking=None,
        v2=False):
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


    Out:
        model: trained model (of a a-priori unknown architecture)
        mb_losses_total (list): list containing the minibatch losses during the adaptive training loop
        test_err_list (list): list of test errors after each "step 1"-training on a model
        test_err_list2 (list): list of test errors computed also during the training

    '''

    if type(epochs) == int:
        epochs = (iters+1)*[epochs]
    if len(epochs) < iters+1:  # if not enough epoch numbers are given, they will be filled with the last entry
        diff = iters+1-len(epochs)
        epochs = epochs + diff * [epochs[-1]]

    mb_losses_total = []
    test_err_list = []
    test_err_list2 = []
    lr = lr_init

    for k in range(iters):
        # iterate on current net
        print(f'starting on {k+1}. net !')
        print(model)
        # build optimizer
        if optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr)

        # build lr scheduler
        if lrschedule_type == 'StepLR':
            step_size = lrscheduler_args['step_size']
            gamma = lrscheduler_args['gamma']
            lrscheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma)

        # train
        mb_losses1, lr_end_lastloop, test_error_l2 = train(
            model, train_dataloader, epochs[k], optimizer, lrscheduler, wanted_testerror=wanted_test_error,
            start_with_backtracking=start_with_backtracking, check_testerror_between=check_testerror_between,
            test_dataloader=test_dataloader, print_param_flag=print_param_flag)

        # train classically until stalling
        mb_losses_total = mb_losses_total + mb_losses1

        test_err_list2 = test_err_list2+test_error_l2

        # test error check if good, break, if not, write in list
        curr_test_err = check_testerror(test_dataloader, model)
        test_err_list.append(curr_test_err)
        print(f'Test error after first step of loop {k+1} is {curr_test_err}!')
        if curr_test_err <= wanted_test_error:
            print(f'The final model has the architecture: {model}')
            return model, mb_losses_total, test_err_list, test_err_list2

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

        # select new model based on the shadow prices
        # insert one frozen layer and unfreeze, delete other frozen layers
        model, kwargs_net, new_child = select_new_model(
            free_norms, freezed_norms, model=model_tmp, freezed=freezed, kwargs_net=kwargs_net_tmp, mode=mode,
            _type=kwargs_net['type'], v2=v2)

        lr = decrease_after_li * lr_end_lastloop  # decrease lr for next loop

    print(f'starting on {k+2}. net!')
    print(model)

    # build optimizer
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr)

    # build lr scheduler
    if lrschedule_type == 'StepLR':
        step_size = lrscheduler_args['step_size']
        gamma = lrscheduler_args['gamma']
        lrscheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)

    mb_losseslast, lr, test_error_l2 = train(model, train_dataloader, epochs[k+1], optimizer, lrscheduler,
                                             wanted_testerror=wanted_test_error,
                                             start_with_backtracking=start_with_backtracking,
                                             check_testerror_between=check_testerror_between,
                                             test_dataloader=test_dataloader, print_param_flag=print_param_flag)
    # after the last li train again

    mb_losses_total = mb_losses_total + mb_losseslast

    test_err_list2 = test_err_list2 + test_error_l2
    curr_test_err = check_testerror(test_dataloader, model)
    test_err_list.append(curr_test_err)

    print(f'Test error of loop {k+2} is {curr_test_err}!')
    print(f'The final model has the architecture: {model}')

    return model, mb_losses_total, test_err_list, test_err_list2
