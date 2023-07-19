import torch
import backtracking


def train(model, train_dataloader, epochs, optimizer, scheduler, wanted_testerror=0.,
          start_with_backtracking=None,
          check_testerror_between=None,
          test_dataloader=None,
          print_param_flag=False):
    '''
    implements (optionally) frozen-parameter constrained training for a feedforward net.

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
    if start_with_backtracking is not None:
        # no of epochs in the beginning where we use backtracking
        backtr_max = start_with_backtracking

    mb_losses = []
    test_err_list = []

    # loop over epochs
    for e in range(epochs):
        # print(f'epoch number {e+1}')
        for X, y in train_dataloader:
            model.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(model(X), y)
            loss.backward()

            # ggf print loss

            mb_losses.append(loss.item())  # store minibatch losses in a list

            if print_param_flag and e <= 10:
                with torch.no_grad():
                    # print(f'at iterate {e}, the weight gradients are:')
                    for p in model.parameters():
                        print(' parameter gradient')
                        print((p.grad).sum())

            if start_with_backtracking is not None and e <= backtr_max:
                lr_btr = backtracking.simple_backtracking(
                    model, 1., diminishing_factor=0.9, curr_loss=loss, x=X, y=y, max_it=10)
                optimizer.param_groups[0]['lr'] = lr_btr

            optimizer.step()

        scheduler.step()

        if check_testerror_between is not None:
            if e % check_testerror_between == 0:
                test_err = check_testerror(test_dataloader, model)
                test_err_list.append(test_err)
                print(f'(dataset)error at epoch {e} is {test_err}')
                if test_err <= wanted_testerror:
                    return mb_losses, optimizer.param_groups[0]['lr'], test_err_list

    return mb_losses, optimizer.param_groups[0]['lr'], test_err_list


def check_testerror(test_dataloader, model):
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
    no_data = test_dataloader.batch_size
    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)

            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

        correct /= no_data
        test_err = 100-100*correct

    return test_err
