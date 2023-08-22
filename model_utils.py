import torch
from activation_functions import TanhLU_shifted
from nets import ResBlock1


@torch.no_grad()
def freeze_params(model, act_fun, old_model, _type='fwd', v2=False):
    '''
    freezes every second propagation, initializes them with identities and zeros and writes all frozen parameters
    in one list.

    Args:
        model: feedforward or resnet2 network which should be trained. It must use activation functions,
        which support inserting identity layers, most popular: ReLU
        act_fun: activation function of the model
        old_model: model, before any identity layers are inserted
        _type: type of the model
        v2 (default False): determines for 2weight resnets whether the init strategy with zeros in the outer or inner weight is taken.
        v2 corresponds to the more restrictive version where the inner weight is initialized with a zero matrix.


    Out:
        freezed: list which contains the frozen parameters of the model. The parameters of model are now initialized such that it resembles the function of old_model
    '''
    freezed = []

    if _type == 'fwd':
        # print('type fwd is called in freezing process!!')
        k = 0
        is_weight = True
        # Freezing and initialization for relu ###########################################################
        if act_fun is torch.nn.ReLU:  # if you want to use the inexact shift for tanhlushifted insert it here
            # and uncomment the next section for tanhlu
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

        # freezing and initialization for tanhlu shifted #######################################################
        if act_fun is TanhLU_shifted:
            for p in model.parameters():
                if k % 2 == 1:  # es startet mit k=0 und is_weight=true
                    if is_weight:  # weight
                        # Setze auf Identität und friere ein
                        # initialize every second propagation with the identity matrix for the weight matrix
                        p.copy_(torch.diag_embed(torch.ones_like(p[0])))
                        # write frozen parameters in the 'freezed' list
                        freezed.append(p)
                        is_weight = False
                        continue

                    else:  # bias
                        p.copy_(torch.ones_like(p[0]))
                        a = 0.01  # this uses a based on the glued point in tanhlushifted
                        b = 0  # because image of tanhlu are positive numbers
                        b_full = torch.mul(torch.tensor(
                            [float(a-b)]), torch.ones_like(p))  # used for init of the next weight
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
                        # wir zählen diesen parameter nicht als gefreezed
                        # weil er noch alle freiheitsgrade hat und nur geshiftet ist
                        is_weight = True
                        k += 1

                else:  # leave the other weights and biases unchanged
                    if not is_weight:
                        k += 1
                    is_weight = not is_weight

    if _type == 'res2':
        v1 = not v2
        # the following two lines determine whether the idenitity or the weight matrix before are used
        # for the initialization of the inner weight
        init_with_weight_before = False
        init_with_identity = True
        scale = 0.8  # scaling of the identity matrix

        # start freezing and initializing
        i = 0
        k = 0
        no = number_of_parameters_of_model(model)
        if v1:  # placeholder for all activation functions

            W1_list = []
            # check if old model has already at least one resblock
            for child in old_model.children():
                if isinstance(child, ResBlock1):
                    init_with_weight_before = not init_with_identity
                    W1_list.append(child.l1.weight.data)
                # if yes save all already inner weight values and set flag to yes

            for p in model.parameters():
                if i == 0 or i == 1:  # parameters of linear layer at beginning
                    # do nothing
                    i += 1
                    continue
                elif i == no-2 or i == no-1:  # parameters of linear layer at the end
                    # do nothing
                    i += 1
                    continue
                elif i % 3 == 2 and i % 6 == 2:
                    # freeze and init innner weight
                    if not init_with_weight_before:
                        p.copy_(torch.diag_embed(scale*torch.ones_like(p[0])))
                    if init_with_weight_before:
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

        if v2:  # works only for activation functions where 0 is a fixed point of it and not of its derivative!
            # relu does not work for example

            W2_list = []
            # check if old model has already at least one resblock
            for child in old_model.children():
                if isinstance(child, ResBlock1):
                    init_with_weight_before = not init_with_identity
                    W2_list.append(child.l2.weight.data)
                # if yes save all already inner weight values and set flag to yes

            for p in model.parameters():
                if i == 0 or i == 1:  # parameters of linear layer at beginning
                    # do nothing
                    i += 1
                    continue
                elif i == no-2 or i == no-1:  # parameters of linear layer at the end
                    i += 1
                    continue
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
                    if not init_with_weight_before:
                        p.copy_(torch.diag_embed(scale*torch.ones_like(p[0])))
                    if init_with_weight_before:
                        p.copy_(W2_list[k])
                        k += 1
                    freezed.append(p)
                    i += 1
                    continue
                else:
                    i += 1
                    continue

    if _type == 'res1':
        i = 0
        no = number_of_parameters_of_model(model)
        zero_fixpoint=True
        is_weight = True
        # Freezing and initialization for relu ###########################################################
        if zero_fixpoint: 
            for p in model.parameters():  # iterate over all model parameters
                if i == 0 or i == 1:  # parameters of linear layer at beginning
                    # do nothing
                    i += 1
                    continue
                elif i == no-2 or i == no-1:  # parameters of linear layer at the end
                    # do nothing
                    i += 1
                    continue
                elif i%2==0 and i%4==2: # new weight 
                    # first new layer is inserted directly after the first linear layer
                    p.mul_(0.)
                    freezed.append(p)
                    i += 1
                    continue
                elif i%2==1 and i%4==3:
                    p.mul_(0.)
                    freezed.append(p)
                    i += 1
                    continue
                else:
                    i += 1
                    continue
        # print(f'freezed parameter shapes{[f.shape for f in freezed]}')
    return freezed


def freeze_old_params(model, new_child):
    '''
    freezes old parameters and lets only the new layer unfrozen.
    '''
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


def _number_of_frozen_parameters_collected(model, freezed):
    '''
     counts total number/dimension of frozen parameters given in 'freezed' in the model
    Args:
        model: model from pytorch
        freezed (list): list of the frozen parameters of the model
    Out:
         dim (int): total number of frozen parameters
    '''
    no = 0
    with torch.no_grad():
        for p in model.parameters():
            if _is_freezed(p, freezed):
                no += 1
    return no


def number_of_free_parameters_collected(model, freezed):
    '''
     counts total number/dimension of not-frozen parameters given in the model
    Args:
        model: model from pytorch
        freezed (list): list of the frozen parameters of the model
    Out:
         dim (int): total number of not-frozen parameters
    '''
    no = 0
    with torch.no_grad():
        for p in model.parameters():
            if not _is_freezed(p, freezed):
                no += 1
    return no


def number_of_parameters_of_model(model):
    no = 0
    with torch.no_grad():
        for p in model.parameters():
            no += 1
    return no
