import torch
from model_utils import _is_freezed, _number_of_params
from model_utils import number_of_free_parameters_collected, _number_of_frozen_parameters_collected


def calculate_shadowprices_minibatch(train_dataloader, model, freezed):
    '''
    calculates gradient entries of the unfrozen and frozen parameters at the current point without a gradient update.
    We look at the squared frobenius norm of the shadow price wrt each parameter scaled by the number of entries
    in the parameter. If the dataloader has minibatches stpred, the shadow prices are averaged over the minibatches
    (without updating the model in this epoch)

    Args:
        train_dataloader: pytorch iterable dataloader which contains the training data
        model: model for which we calculate the gradient entries
        freezed: list of frozen parameters in model

    Out:
        not_frozen: list of unfrozen gradient entries averaged as the unfrozen ones.
        frozen: list of averaged frozen sensitivities (=frozen parameter gradients)
        loss_value: current value of loss function

    '''
    not_frozen_big = []
    frozen_big = []
    with torch.no_grad():
        for p in model.parameters():
            if not _is_freezed(p, freezed):
                not_frozen_big.append(torch.zeros_like(p.data))
            else:
                frozen_big.append(torch.zeros_like(p.data))



    
    loss_values = []

    for X, y in train_dataloader:
        model.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(
            model(X), y)
        loss.backward()
        loss_values.append(loss.item())

        k = 0  # counts the number of frozen parameters
        kk = 0  # counts the number of free parameters
        with torch.no_grad():
            for p in model.parameters():  # iterate over all model parameters
                # if parameter is not frozen, append gradient to list
                if not _is_freezed(p, freezed):
                    #old not_frozen[kk] += torch.sum(torch.square(p.grad))#/_number_of_params(p) ## NEW: if unscaled, comment out the division by _number_of_params(p)
                    not_frozen_big[kk] += p.grad
                    kk += 1
                # if parameters is frozen, append one averaged gradient value to list
                if _is_freezed(p, freezed):
                    # print(p.grad) # uncomment if you want to see all shadow price values of the model parameters
                    # and not just an average
                    #old frozen[k] += torch.sum(torch.square(p.grad))#/_number_of_params(p) ## NEW: if unscaled, comment out the division by _number_of_params(p)
                    frozen_big[k] += p.grad
                    k += 1

    # scale the frozen and unfrozen lists with by the numbers of batches
    scale = 1/len(loss_values)
    # square and sum up the p.grads over all its from big into frozen, not_frozen
    frozen = [torch.sum(torch.square(scale*p)) for p in frozen_big]
    not_frozen = [torch.sum(torch.square(scale*p)) for p in not_frozen_big]

    return not_frozen, frozen, loss_values


# def calculate_shadowprices_fullbatch(model, full_training_data_in, full_training_data_out, freezed):
#     '''
#     calculates gradient entries of the unfrozen and frozen parameters at the current point without a gradient update.
#     We look at the squared frobenius norm of the shadow price wrt each parameter scaled by the number of entries
#     in the parameter.

#     Args:
#         model: model for which we calculate the gradient entries
#         full_training_data_in: inputs of the training data
#         full_training_data_out: outputs of the training data
#         freezed: list of frozen parameters in model

#     Out:
#         not_frozen: list of unfrozen gradient entries averaged as the unfrozen ones.
#         frozen: list of averaged frozen sensitivities (=frozen parameter gradients)
#         loss_value: current value of loss function

#     '''

#     not_frozen = []
#     frozen = []
#     model.zero_grad()
#     loss = torch.nn.CrossEntropyLoss()(
#         model(full_training_data_in), full_training_data_out)
#     loss.backward()

#     with torch.no_grad():
#         for p in model.parameters():  # iterate over all model parameters
#             # if parameter is not frozen, append gradient to list
#             if not _is_freezed(p, freezed):
#                 not_frozen.append(
#                     [torch.sum(torch.square(torch.abs(p.grad)))/_number_of_params(p)])
#             # if parameters is frozen, append one averaged gradient value to list
#             if _is_freezed(p, freezed):
#                 # print(p.grad) # uncomment if you want to see all shadow price values of the model parameters
#                 # and not just an average
#                 frozen.append(
#                     [torch.sum(torch.square(torch.abs(p.grad)))/_number_of_params(p)])

#     loss_value = [loss.item()]
#     return not_frozen, frozen, loss_value