import torch
from model_utils import _is_freezed, _number_of_params
from calc_inexact_term import calculate_inexact_term


def calculate_shadowprices_for_one_gradient(model, loss_fn, full_training_data_in, full_training_data_out, freezed, calc_inexact_term=False, kwargs_net=None):
    '''
    calculates gradient entries of the unfrozen and frozen parameters at the current point without a gradient update. The gradient of the unfrozen parameters are unchanged, the gradient values of the frozen parameters are averaged (sum of abs of grad entries) and scaled per parameter (divided by number dimension of parameter).

    Args:
        model: model for which we calculate the gradient entries
        loss_fn: should be CEloss
        full_training_data_in: inputs of the training data
        full_training_data_out: outputs of the training data
        freezed: list of frozen parameters in model

    Out:
        not_frozen: list of unfrozen gradient entries (parameterwise)
        frozen: list of averaged frozen parameter gradients
        loss_value: current value of loss function 

    '''

    not_frozen = []
    frozen = []  # [[] for _ in freezed]
    complete_grad = []
    model.zero_grad()
    loss = loss_fn(model(full_training_data_in), full_training_data_out)
    loss.backward()

    with torch.no_grad():
        for p in model.parameters():  # iterate over all model parameters
            # if parameter is not frozen, append gradient to list
            if not _is_freezed(p, freezed):
                not_frozen.append(p.grad)
            # if parameters is frozen, append one averaged gradient value to list
            if _is_freezed(p, freezed):
                #print(p.grad) # uncomment if you want to see all shadow price values of the model parameters and not just an average
                frozen.append([torch.sum(p.grad)/_number_of_params(p)])
            complete_grad.append(p.grad)

    loss_value = [loss.item()]
    if calc_inexact_term:
        inex_term = calculate_inexact_term(
            model, kwargs_net, full_training_data_in, full_training_data_out, freezed)
        return not_frozen, frozen, loss_value, inex_term

    return not_frozen, frozen, loss_value
