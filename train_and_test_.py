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
        train_dataloader: iterable from pytorch containing the training data
        epochs (int): number of epochs, in which we want to train equality constrained. default 1
        optimizer: optimizer, e.g. torch.nn.optim.SGD
        scheduler: lr scheduler, e.g. torch.nn.optim.lr_scheduler.StepLR
        wanted_testerror: between 100 and 0 indicating which testerror is sufficient to stop.
        start_with_backtracking (None or int): if None, there is  no backtracking performed, if it is an integer k ,
            then for the first k epochs, backtracking is performed after the layer insertion.
        check_testerror_between (None or int): if none, the testerror is noct checked while training on a model.
            If an integer is specified, e.g. k=1, the after k epochs, the testerror is checked during training on one model.
        test_dataloader: iterable from pytorch containing the test data
        print_param_flag (default False): if True, prints the parameter gradients for the first 10 epochs

    Out:
        mb_losses (list): list of all minibatch losses during training
        lr: current lr used at end of training
        test_err_list (list): list which contain the testerrors during training if computed
        exit_flag: 0 or 1. 1 indicates that wanted testerror was reached during training

    '''
    exit_flag = 0
    if start_with_backtracking is not None:
        # no of epochs in the beginning where we use backtracking
        backtr_max = start_with_backtracking

    mb_losses = []
    test_err_list = []

    # loop over epochs
    for e in range(epochs):
        # print(f'epoch number {e+1}')
        for batch, (X, y) in enumerate(train_dataloader):
            model.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(model(X), y)
            loss.backward()

            # ggf print loss
            if batch == 0:
                print('mbloss: ', loss.item())

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
                    exit_flag = 1
                    return mb_losses, optimizer.param_groups[0]['lr'], test_err_list, exit_flag

    return mb_losses, optimizer.param_groups[0]['lr'], test_err_list, exit_flag


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
