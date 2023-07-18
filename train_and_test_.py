from extract_node_info_automatic import get_timestamp
# from gif_maker import make_gif
import os
import torch
from slucks import plot_decision_boundary
from random import sample
from model_utils import _is_freezed, _number_of_params
import backtracking


def train_model_on_spiral_freezed_params(model, loss_fn, training_data_in, training_data_out, stall_detector, epochs=1,
                                         bs=10, lr=.1, use_adaptive_lr=False, freezed=None,
                                         start_with_backtracking=False, save_decbound_after_li=False,
                                         check_testerror_between=None, test_data_x=None, test_data_y=None,
                                         wanted_testerror=0.,
                                         print_param_flag=False):
    '''
    implements (optopnally) frozen-parameter constrained training for a feedforward net.

    Args:
        model: model with frozen (or no frozen) parameters
        loss_fn: loss function
        training_data_in: torch.tensor which holds the training data features
        training_data_out: torch.tensor which holds the training data labels
        stall_detector: object of class stall_detector, which handles the stalling checking
        epochs (int): number of epochs, in which we want to train equality constrained. default 1
        bs (int): batch size, default 10
        lr (float): learning rate, default 0.1
        use_adaptive_lr (bool): boolean indicating whether we use a fixed lr or the adaptive version
        by Frederik Köhne. default False
        freezed (list): list of frozen parameters in the model.
        if none are frozen use freezed=None (which is also the default)

    Out:
        norms (list): a list containing the scaled gradient 2-norms for each minibatch computation
        freezed_norms (list of lists): each list in the list represents one frozen model parameter
        (so weight or bias). it contains the scaled gradient norm wrt the specific parameter
        for all minibatches
        mb_losses (list): list of all minibatch losses during training

    '''
    if not freezed:
        freezed = []
    backtr_max = 0  # no of epochs in the beginning where we use backtracking

    prevent_net_blowup = False
    if prevent_net_blowup:
        print('Attention: with this choice of the stalling, the net can be set back to the previous net' +
              'when no descent is given after the sleep period!')

    no_steps_movie = 100
    if save_decbound_after_li:
        # create directory
        # Directory
        directory = "layer_insertion_at_time" + get_timestamp()
        # print(f'cwd: {os.getcwd()}')
        # Parent Directory path
        parent_dir = "figs/dec_bound_movies"  # "D:/Pycharm projects/"

        # Path
        path = os.path.join(parent_dir, directory)
        # Create the directory
        # 'GeeksForGeeks' in
        # '/home / User / Documents'
        os.mkdir(path)

    data_x = training_data_in  # training input data
    data_y = training_data_out  # training output data

    if use_adaptive_lr:
        omega = 10
    discont = .995
    omega_min = 1e-5

    dataset_size = len(data_x)
    mb_losses = []
    norms = []
    replaced_norms = [[] for _ in freezed]
    replaced2_norms = []
    freezed_norms = [[] for _ in freezed]
    test_err_list = []

    # this computes the number of all trainable (so not frozen) parameters, this is needed for the scaling
    trainable_params = 0
    for p in model.parameters():
        if not _is_freezed(p, freezed):
            trainable_params += _number_of_params(p)

    bs = dataset_size  # uncomment for GD

    # loop over epochs
    for e in range(epochs):
        # print(f'epoch number {e+1}')
        samples_seen = 0
        while samples_seen < dataset_size:
            # sample:

            batch = sample(range(0, dataset_size), bs)  # generate batch

            if save_decbound_after_li and e < no_steps_movie:

                plot_decision_boundary(
                    model, data_x[batch], data_y[batch], save_plot=path+"/"+str(e)+".png")

            model.zero_grad()  # set gradients to zero (must be done after each backward() call!)

            # compute minibtach loss
            loss = loss_fn(model(data_x[batch]), data_y[batch])

            loss.backward()  # backpropagation

            if e % 100 == 0:
                print(f' at epoch {e} loss is {loss.item()}')

            mb_losses.append(loss.item())  # store minibatch losses in a list

            if check_testerror_between is not None:
                if e % check_testerror_between == 0:
                    test_err = check_testerror(test_data_x, test_data_y, model)
                    test_err_list.append(test_err)
                    print(f'(dataset)error at epoch {e} is {test_err}')
                    if test_err <= wanted_testerror:
                        return replaced_norms, freezed_norms, mb_losses, lr, test_err_list

            if print_param_flag and e <= 100:
                with torch.no_grad():
                    # print(f'at iterate {e}, the weight gradients are:')
                    for p in model.parameters():
                        if _is_freezed(p, freezed):
                            print(' ')  # platzhalter
                        #    print('frozen parameter gradient')
                        # print((p.grad**2).sum())

            # print(f'loss: {loss.item()}')

            # update non-frozen parameters with a gradient update
            with torch.no_grad():
                step_norm_sq = torch.tensor(0.)
                if start_with_backtracking and e < backtr_max:
                    # lr = backtracking.simple_backtracking(
                    #    model, lr, 0.9, mb_losses[-1], loss_fn, training_data_in, training_data_out, max_it=20)
                    # # using the last lr as first backtracking lr
                    lr = backtracking.simple_backtracking(
                        model, .1, 0.5, mb_losses[-1], loss_fn, training_data_in, training_data_out, max_it=20)
                    # using .1 as first backtracking lr
                    omega = 1/lr  # give good starting point for adaptive
                else:
                    for p in model.parameters():
                        # print(p.grad)
                        if e <= 50:
                            # for checking symmetry problems only:
                            print(
                                f'epoch {e} values of model parameters: {p}')
                        if not _is_freezed(p, freezed):
                            p.add_(p.grad, alpha=-lr)
                            # compute squared norm of gradient
                            step_norm_sq.add_((p.grad ** 2).sum())
                            # print(f'not frozen: {p.grad}')
                            replaced2_norms.append(p.grad)

                    norms.append(float(step_norm_sq) / trainable_params)

                # give metric to stallingdetector
                stall_detector.tell_metric(float(loss), bs)

                for replaced_list, freezed_norm_list, p in zip(replaced_norms, freezed_norms, freezed):
                    # print(f'frozen: {p.grad}')
                    # compute scaled squared gradient norms of the frozen parameters
                    freezed_norm_list.append(
                        (p.grad ** 2).sum() / _number_of_params(p))
                    replaced_list.append(p.grad)

                if use_adaptive_lr and e > backtr_max:
                    loss_new = loss_fn(model(data_x[batch]), data_y[batch])

                    omega_new = 2 * (loss_new - loss + lr *
                                     step_norm_sq) / (lr ** 2 * step_norm_sq)
                    if omega_new < omega_min:
                        omega_new = omega_min
                    omega = discont * omega + (1 - discont) * omega_new
                    lr = discont * lr + (1 - discont) * 1/omega

            # model.zero_grad() # set gradients to zero (must be done after each backward() call!)
            samples_seen += bs

        # print(f'Finished epoch {e + 1}, last loss = {loss}, lr = {lr}')
        stall_detector.end_epoch()
        # print(f'stall detector best value: {stall_detector.best_value}')
        # print(f'curr epoch mean: {stall_detector.curr_epoch_mean}')
        if stall_detector.bad_epoch_counter == 0:
            stall_detector.update_best_params(model)
        # print(f'Bad Epoch Count = {stall_detector.bad_epoch_counter}')

        if stall_detector.is_stalling or e == epochs-1:
            if stall_detector.is_stalling:
                # set back model parameters to last good epoch
                # torch.nn.utils.vector_to_parameters(
                # stall_detector.current_best_parameter, model.parameters())
                model = stall_detector.current_best_model
                print(
                    f'Attention: due to bad loss we reset the model to {model}')
            print(f'Finished epoch {e + 1}, last loss = {loss}, lr = {lr}')
            break
    # print(f'cwd bef make_gif: {os.getcwd()}')
    # make_gif(path)

    if prevent_net_blowup:
        stall_detector.reset_with_old_info()
    else:
        stall_detector.reset()
    return replaced_norms, freezed_norms, mb_losses, lr, test_err_list


def check_testerror(test_data_x, test_data_y, model):
    '''
    computes the test error of a model wrt the given test data. the test error is returned between
    100(= all wrong) and 0(=all correct).

    Args:
        test_data_x: torch.tensor which holds the test inputs
        test_data_y: torch.tensor which holds the test outputs
        model: pytorch model

    Out:
        test_err (float): test error between 100 and 0.
    '''
    correct = 0
    no_data = len(test_data_y)
    with torch.no_grad():
        for X, y in zip(test_data_x, test_data_y):
            # print(X)
            pred = model(X)
            # y.argmax(1)
            correct += (pred.argmax() == y).type(torch.float).sum().item()

        correct /= no_data
        test_err = 100-100*correct

    return test_err
