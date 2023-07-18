import torch
import numpy as np
from activation_functions import TanhLU_shifted, tanh_test
from extract_node_info_automatic import save_node_values
from net import ResBlock1


@torch.no_grad()
def freeze_params(model, act_fun, old_model, training_data, _type='fwd'):
    '''
    freezes every second propagation, initializes them with identities and zeros and writes all frozen parameters
    in one list.

    Args:
        model: feedforward or resnet2 network which should be trained. It must use activation functions,
        which support inserting identity layers, most popular: ReLU

    Out:
        list which contains the frozen parameters of the model
    '''
    freezed = []

    if _type == 'fwd':
        # print('type fwd is called in freezing process!!')
        k = 0
        is_weight = True
        if act_fun is torch.nn.ReLU or act_fun is tanh_test:
            for p in model.parameters():  # iterate over all model parameters
                if k % 2 == 1:
                    if is_weight:
                        # Setze auf Identität und friere ein
                        # initialize every second propagation with the identity matrix for the weight matrix
                        p.copy_(torch.diag_embed(torch.ones_like(p[0])))
                        # write frozen parameters in the 'freezed' list
                        freezed.append(p)
                        is_weight = False
                        continue

                    else:
                        p.mul_(0.)  # initialize every second bias with zeros
                        # write frozen parameters in the 'freezed' list
                        freezed.append(p)
                        is_weight = True
                        k += 1
                        continue

                else:
                    # leave the other weights and biases unchanged
                    if not is_weight:
                        k += 1
                    is_weight = not is_weight

        if act_fun is TanhLU_shifted:
            exact_shift_expensive = False
            exact_shift_cheap = True
            inexact_shift = False
            if exact_shift_expensive:
                preds_old, hidden_nodes_old = save_node_values(
                    old_model, training_data, w_actfun=False)

            for p in model.parameters():
                # print(f'p is {p}')
                # print(hidden_nodes_old[k].shape)
                if k % 2 == 1:
                    init_w_id = False
                    init_random = True
                    if is_weight:
                        # Setze auf Identität und friere ein
                        if init_w_id:
                            # initialize every second propagation with the identity matrix for the weight matrix
                            p.copy_(torch.diag_embed(torch.ones_like(p[0])))
                        if init_random:
                            # initialize inner weight with e.g. xavier initialization
                            p.copy_(p[0])
                        # write frozen parameters in the 'freezed' list
                        freezed.append(p)
                        is_weight = False
                        continue

                    else:
                        # print(' compute minimum of the follwoing nodes')
                        # print(hidden_nodes_old[int((k-1)/2)].shape)
                        p.copy_(torch.ones_like(p[0]))
                        if exact_shift_expensive:
                            a = 0.01
                            b = np.min(hidden_nodes_old[int((k-1)/2)])
                        if inexact_shift:  # this is the same init as for relu
                            a = 0
                            b = 0
                        if exact_shift_cheap:
                            a = 0.01
                            b = 0
                        b_full = torch.mul(torch.tensor(
                            [float(a-b)]), torch.ones_like(p))
                        # b for each entry where b=1-min knotenwerte aller td davor
                        p.mul_(a-b)
                        # write frozen parameters in the 'freezed' list
                        freezed.append(p)
                        is_weight = True
                        k += 1
                        continue

                if k % 2 == 0 and k != 0:
                    if is_weight:
                        # calc W2*b
                        W2b = torch.matmul(p[0], b_full)
                        is_weight = False
                        continue

                    else:
                        p.sub_(W2b)
                        # freezed.append(p) #????? probably not? es müsste nur der extra term gefreezed werden
                        is_weight = True
                        k += 1

                else:  # leave the other weights and biases unchanged
                    if not is_weight:
                        k += 1
                    is_weight = not is_weight

    if _type == 'res2':
        # v1/v2 must be changed here and in model_selection.py
        v1 = True
        v2 = not v1
        init_flag = False
        W1_list = []
        # check if old model has already at least one resblock
        for child in old_model.children():
            if isinstance(child, ResBlock1):
                init_flag = False
                W1_list.append(child.l1.weight.data)
        # if yes save all already inner weight values and set flag to yes

        # start freezing and initializing
        i = 0
        k = 0
        no = number_of_parameters_of_model(model)
        if v1:  # placeholder for activation function which map 0 to 0
            for p in model.parameters():
                # print(f'{i} shape pf current p: {p.shape}')
                if i == 0 or i == 1:  # parameters of linear layer at beginning
                    # do nothing
                    i += 1
                    continue
                elif i == no-2 or i == no-1:  # parameters of linear layer at the end
                    i += 1
                    continue
                # elif i == 2:
                #     # freeze and init first inner weight
                #     p.copy_(torch.diag_embed(torch.ones_like(p[0])))
                #     freezed.append(p)
                #     i += 1
                #     continue
                elif i % 3 == 2 and i % 6 == 2:
                    # freeze and init innner weight
                    if not init_flag:
                        p.copy_(torch.diag_embed(0.8*torch.ones_like(p[0])))
                    if init_flag:
                        p.copy_(W1_list[k])
                        k += 1
                    freezed.append(p)
                    i += 1
                    continue
                elif i % 3 == 0 and i % 6 == 3:
                    # freeze and init bias
                    p.mul_(0.)
                    freezed.append(p)
                    i += 1
                    continue
                elif i % 3 == 1 and i % 6 == 4:
                    # freeze and init outer weight
                    p.mul_(0.)
                    freezed.append(p)
                    i += 1
                    continue
                else:
                    i += 1
                    continue

        if v2:
            for p in model.parameters():
                # print(f'{i} shape pf current p: {p.shape}')
                if i == 0 or i == 1:  # parameters of linear layer at beginning
                    # do nothing
                    i += 1
                    continue
                elif i == no-2 or i == no-1:  # parameters of linear layer at the end
                    i += 1
                    continue
                # elif i == 2:
                #     # freeze and init first inner weight
                #     p.copy_(torch.diag_embed(torch.ones_like(p[0])))
                #     freezed.append(p)
                #     i += 1
                #     continue
                elif i % 3 == 2 and i % 6 == 2:
                    # freeze and init innner weight
                    p.mul_(0.)
                    freezed.append(p)
                    i += 1
                    continue
                elif i % 3 == 0 and i % 6 == 3:
                    # freeze and init bias
                    p.mul_(0.)
                    freezed.append(p)
                    i += 1
                    continue
                elif i % 3 == 1 and i % 6 == 4:
                    # freeze and init outer weight
                    if True:  # not init_flag:
                        p.copy_(torch.diag_embed(1.*torch.ones_like(p[0])))
                    if False:  # init_flag:
                        p.copy_(W1_list[k])
                        k += 1
                    freezed.append(p)
                    i += 1
                    continue
                else:
                    i += 1
                    continue

        print(f'freezed parameter shapes{[f.shape for f in freezed]}')
    return freezed




def freeze_old_params(model, new_child):
    old_params = []
    for child in model.children():
        if child is not new_child:
            old_params.extend(child.parameters())

    return old_params


def list_of_parameters(model):
    '''
    not used for now.
    '''
    list_p = torch.empty(0)
    for p in model.parameters():
        list_p = torch.stack((list_p, p))
    return list_p


def _is_freezed(p, freezed):
    '''
    checks if the parameter p is in the list of frozen parameters 'freezed'
    Args:
        p: a model parameter (p in model.parameters())
        freezed (list): list of the frozen parameters of the model
    Out:
         boolean
    '''
    for freezed_p in freezed:
        # print(p.shape, freezed_p.shape)
        try:
            if p is freezed_p:
                return True

        except RuntimeError as m:
            print(m)
            continue
    return False


def _number_of_params(p):
    '''
    computes the dimension of a parameter (vectorized for matrices)

    Args:
        p: model parameter from which we want to know the dimension

    Out:
        integer, which gives the dimension of the parameter'''
    res = 1
    for s in p.shape:
        res *= s
    return res


def _number_of_frozen_parameters(model, freezed):
    '''
     counts total number/dimension of frozen parameters given in 'freezed' in the model
    Args:
        model: model from pytorch
        freezed (list): list of the frozen parameters of the model
    Out:
         dim (int): total number of frozen parameters
    '''
    dim = 0
    with torch.no_grad():
        for p in model.parameters():
            if _is_freezed(p, freezed):
                dim += _number_of_params(p)
    return dim


def number_of_free_parameters(model, freezed):
    '''
     counts total number/dimension of not-frozen parameters given in the model
    Args:
        model: model from pytorch
        freezed (list): list of the frozen parameters of the model
    Out:
         dim (int): total number of not-frozen parameters
    '''
    dim = 0
    with torch.no_grad():
        for p in model.parameters():
            if not _is_freezed(p, freezed):
                dim += _number_of_params(p)
    return dim


def number_of_parameters_of_model(model):
    no = 0
    with torch.no_grad():
        for p in model.parameters():
            no += 1
    return no
