import torch
import time 
import backtracking
import os
import numpy as np

from heatmap import save_weightgrads_heatmap, save_weightvalues_heatmap


def train(model, train_dataloader, epochs, optimizer, scheduler, wanted_testerror=0.,
          start_with_backtracking=None,
          check_testerror_between=None,
          test_dataloader=None,
          print_param_flag=False,
          save_grad_norms=False,
          use_adaptive_lr=False,
          loss_fn=torch.nn.CrossEntropyLoss(),
          test_mode = '01',
          save_heatmaps=False):
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
        save_grad_norms: (bool) default False. If True, saves the layerwise averaged squared norm of the gradient in each step.
        use_adaptive_lr: (bool) default False. If True, uses adaptive lr for the training
        loss_fn: loss function, default torch.nn.CrossEntropyLoss()
        test_mode: '01' or 'mse'. If '01', the test error is computed as classification error, if 'mse', the test error is computed as mse.
        save_heatmaps: (bool) default False. If True, saves the heatmaps of the gradients of the weights in each step.

    Out:
        mb_losses (list): list of all minibatch losses during training
        lr: current lr used at end of training
        test_err_list (list): list which contain the testerrors during training if computed
        exit_flag: 0 or 1. 1 indicates that wanted testerror was reached during training
        grad_norms_layerwise: [] if save_gard_norms is False, otherwise a list of the layerwise averaged gradients for each iteration
        times: list of times of the iterates

    '''
    grad_norms = []
    grad_norms_layerwise = []
    for p in model.parameters():
        grad_norms_layerwise.append([])
    exit_flag = 0
    if start_with_backtracking is not None:
        # no of epochs in the beginning where we use backtracking
        backtr_max = start_with_backtracking

    if save_heatmaps:
        heatmappathgrads=f'heatmaps/grads/grads_{time.time()}/'
        os.mkdir(heatmappathgrads)
        
        heatmappathvals=f'heatmaps/weights/vals_{time.time()}/'
        os.mkdir(heatmappathvals)
        

    mb_losses = []
    test_err_list = []
    times = [0]

    test_err = check_testerror(test_dataloader, model, test_mode=test_mode)
    
    print(f'test error before training is {test_err}')

    if use_adaptive_lr:
        omega = 10
        discont = .995
        omega_min = 1e-5

    # loop over epochs
    for e in range(epochs):
        tic = time.time()
        # print(f'epoch number {e+1}')
        for batch, (X, y) in enumerate(train_dataloader):
            #print(X.shape)
            #print(y.shape)
            model.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()

            if save_heatmaps:
                # save heatmaps of gradients and values
                #save_weightgrads_heatmap(model, batch, e, path=heatmappathgrads)
                #save_weightvalues_heatmap(model, batch, e, path=heatmappathvals)
                #save weights and grads for heatmapplotting later
                with torch.no_grad():
                    for i, p in enumerate(model.parameters()):
                        filename = f'{heatmappathvals}epoch{e}_batch{batch}_param{i}.txt'
                        np.savetxt(filename, p.data.detach().numpy())
                        filename = f'{heatmappathgrads}epoch{e}_batch{batch}_param{i}.txt'
                        np.savetxt(filename, p.grad.data.detach().numpy())
               


            if save_grad_norms:
                norm = 0
                layer = 0
                lr = optimizer.param_groups[0]['lr']
                for p in model.parameters():
                    grad_norms_layerwise[layer].append(
                        lr*torch.square(p.grad).sum().numpy()) # old was lr after norm
                    layer += 1
                for p in model.parameters():
                    norm += torch.square(p.grad).sum()
                # print(norm)
                grad_norms.append(norm)

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

            if use_adaptive_lr: ##### new beginning
                    with torch.no_grad():
                        step_norm_sq=torch.tensor(0.)
                        for p in model.parameters():
                            step_norm_sq.add_((p.grad ** 2).sum())
                        loss_new = loss_fn(model(X), y)#loss_fn(model(data_x[batch]), data_y[batch])

                        omega_new = 2 * (loss_new - loss + lr *
                                        step_norm_sq) / (lr ** 2 * step_norm_sq)
                        if omega_new < omega_min:
                            omega_new = omega_min
                        omega = discont * omega + (1 - discont) * omega_new
                        lr = discont * lr + (1 - discont) * 1/omega ################ new end
                        optimizer.param_groups[0]['lr'] = lr

        toc = time.time()
        delta = toc-tic
        times.append(delta+ times[-1])
        scheduler.step()

        if check_testerror_between is not None:
            if e % check_testerror_between == 0:
                test_err = check_testerror(test_dataloader, model, test_mode=test_mode)
                test_err_list.append(test_err)
                print(f'test error at epoch {e} is {test_err}')
                if test_err <= wanted_testerror:
                    exit_flag = 1
                    return mb_losses, optimizer.param_groups[0]['lr'], test_err_list, exit_flag, grad_norms_layerwise, times

    return mb_losses, optimizer.param_groups[0]['lr'], test_err_list, exit_flag, grad_norms_layerwise, times


def check_testerror(test_dataloader, model, test_mode='01'):
    '''
    computes the test error of a model wrt the given test data. the test error is returned between
    100(= all wrong) and 0(=all correct).

    Args:
        test_dataloader: pxtorch dataloader of testdata
        model: pytorch model

    Out:
        test_err (float): test error between 100 and 0 in %.
    '''
    if test_mode == '01':
        correct = 0
        #no_data = test_dataloader.batch_size
        i = 0
        with torch.no_grad():
            for X, y in test_dataloader:
                no_data = X.shape[0]
                pred = model(X)

                # dim=1 for image class, 0 for spirals
                correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()
                i += 1

            correct = correct / (i * no_data)
            test_err = 100 - 100 * correct

    elif test_mode == 'mse':
        with torch.no_grad():
            loss_fn = torch.nn.MSELoss()
            test_err = 0
            for X, y in test_dataloader:
                pred = model(X)
                test_err += loss_fn(pred, y).item()
            #test_err = test_err / len(test_dataloader)

    return test_err