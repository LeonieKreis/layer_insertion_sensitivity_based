import torch
import net
from model_utils import freeze_params, _is_freezed


def tmp_net(hidden_layers, dim_hidden_layers, act_fun, _type, model, training_data):
    '''
    builds a new model for the equality constrained training with frozen parameters based on an old/coarse model.

    Args:
        hidden_layers (int): number of hidden layers in the old/coarse model
        dim_hidden_layers (int or list): the width of the hidden layers. If int, all hidden layers have the same width.
        model: old/coarse pytorch model

    Out:
        new_model: newly constructed pytorch model with identity mappings in the new layers
        freezed (list): list of the frozen parameters in new_model
        new_kwargs_net (dict): kwargs of the new_model
        '''
    if _type == 'fwd':
        if isinstance(dim_hidden_layers, int):
            dim_hidden_layers = [dim_hidden_layers] * hidden_layers

        dim_hidden_layers_new = []
        # the width of the hidden layers in the new model stay pairwise the same. e.g. [3,4]->[3,3,4,4]
        for dim in dim_hidden_layers:
            dim_hidden_layers_new += 2 * [dim]

        hidden_layers_new = 2 * hidden_layers  # new depth is 2* old depth

        new_kwargs_net = {'hidden_layers': hidden_layers_new,
                          'dim_hidden_layers': dim_hidden_layers_new,
                          'act_fun': act_fun,
                          'type': _type}

        new_model = net.feed_forward(
            hidden_layers_new, dim_hidden_layers_new, act_fun=act_fun)  # build new model
        # print(f'tmp net {new_model}')
        # freeze and initialize every second layer of the new model
        freezed = freeze_params(new_model, act_fun, model, training_data)

        # the not-frozen layers get initialized with the parameter values from the old/coarse model
        with torch.no_grad():
            old_param_iterator = model.parameters()
            for p_new in new_model.parameters():
                if not _is_freezed(p_new, freezed):
                    p = next(old_param_iterator)
                    p_new.copy_(p)
        return new_model, freezed, new_kwargs_net

    if _type == 'res2':

        if isinstance(dim_hidden_layers, int):
            dim_hidden_layers = [dim_hidden_layers] * hidden_layers

        if hidden_layers == 1:
            hidden_layers_new = 2
            dim_hidden_layers_new = [
                dim_hidden_layers[0], dim_hidden_layers[0]]
        if hidden_layers > 1:
            hidden_layers_new = 2 * hidden_layers - 1  # TODO
            dim_hidden_layers_new = (2*hidden_layers-1) * \
                [dim_hidden_layers[0]]

        new_kwargs_net = {'hidden_layers': hidden_layers_new,
                          'dim_hidden_layers': dim_hidden_layers_new,
                          'act_fun': act_fun,
                          'type': _type}

        new_model = net.two_weight_resnet(
            hidden_layers_new, dim_hidden_layers_new, act_fun=act_fun)  # build new model
        # print(f'tmp net {new_model}')
        # freeze and initialize every second layer of the new model
        freezed = freeze_params(new_model, act_fun, model,
                                training_data, _type='res2')

        # the not-frozen layers get initialized with the parameter values from the old/coarse model
        with torch.no_grad():
            old_param_iterator = model.parameters()
            print(f'len of freezed: {len(freezed)}')
            for p_new in new_model.parameters():
                if not _is_freezed(p_new, freezed):
                    p = next(old_param_iterator)
                    p_new.copy_(p)
        return new_model, freezed, new_kwargs_net


def select_new_model(avg_grad_norm, freezed_norms, model, freezed, kwargs_net, mode='abs max', _type='fwd'):
    '''
    builds new model based on the lagrangemultipliers contained in freezed_norms by some selection procedure. for now,
    only one layer gets chosen which has the biggest scaled gradient (2-)norm in the weight matrix over one epoch.

    Args:
        avg_grad_norm (list): scaled avg gradient norms over 1 epoch (NOT used for now)
        freezed_norms (list of lists): has length=2*no of freezed layers. each two lists in the list represents
        one frozen layer by the sacled norms of the weight and bias, starting from the first weight, first bias,
        second weight,.. and so on. each list contains the scaled gradient norms wrt the parameter for one epoch.
        model: model from the equality constrained training, i.e. the network with half frozen layers
        freezed (list): list of the freezed parameters in model
        mode (string): either 'min' or 'max', 'abs min' or 'abs max' so far, indicating whether the layer with th4
        largest or smallest or absolute value smallest (comparison) or absolute value largest (theory)
        lagrange multipliers are chosen in model selection. default is 'abs max'.

    Out:
        new_model: model which has the selected new layer initilazied with identity and zero but not frozen.
        new_kwargs_net (dict): kwargs of new_model
    '''
    freezed_norms_means = [
    ]  # list which will store the mean (over the minibatches) of the scaled gradient
    # norms of the weight matrices for each frozen layer
    freezed_norms_means_abs = []

    if _type == 'fwd':
        for k, freezed_norm in enumerate(freezed_norms):
            # print('sum of freezed_norm')
            # print(sum(freezed_norm))
            if k % 2 == 0:
                # store mean over epoch for each frozen layer weight
                freezed_norms_means.append(
                    sum(freezed_norm) / len(freezed_norm))
                sumabs = sum(abs(entry) for entry in freezed_norm)
                freezed_norms_means_abs.append(sumabs/len(freezed_norm))
        # print(f'len of freezed norms (layerwise): {len(freezed_norms_means)}')

    if _type == 'res2':
        for k, freezed_norm in enumerate(freezed_norms):
            # print('sum of freezed_norm')
            # print(sum(freezed_norm))

            # v1/ v2 must be changed in this file and model_utils.py
            v1 = True
            v2 = not v1
            if v1:
                weight = 2
            if v2:
                weight = 0
            if k % 3 == weight:
                # store mean over epoch for each frozen layer weight
                freezed_norms_means.append(
                    sum(freezed_norm) / len(freezed_norm))
                sumabs = sum(abs(entry) for entry in freezed_norm)
                freezed_norms_means_abs.append(sumabs/len(freezed_norm))
        # print(f'len of freezed norms (layerwise): {len(freezed_norms_means)}')

    if mode == 'abs max':
        # find index which has maximum absolute mean norm entry #
        max_index = max(range(len(freezed_norms_means_abs)),
                        key=lambda l: freezed_norms_means_abs[l])

        if _type == 'fwd':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 * max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                # check only linear layers (no activation functions)
                if isinstance(child, torch.nn.Linear):
                    # if layer in model is not frozen or is the best layer,
                    # append the linear layer and Relu activation after
                    if not _is_freezed(child.weight, freezed):
                        new_model_children_list.append(child)
                        new_model_children_list.append(kwargs_net['act_fun']())

                    if child.weight is best_layer_weight:
                        child_for_return = child
                        new_model_children_list.append(child)
                        new_model_children_list.append(kwargs_net['act_fun']())

            new_kwargs_net = {'hidden_layers': 0, 'dim_hidden_layers': [
            ], 'act_fun': kwargs_net['act_fun'], 'type': _type}

            for k in range(len(new_model_children_list)-1):  # build kwargs of the new net
                if isinstance(new_model_children_list[k], kwargs_net['act_fun']):
                    new_kwargs_net['hidden_layers'] += 1
                    new_kwargs_net['dim_hidden_layers'].append(
                        new_model_children_list[k - 1].out_features)

        if _type == 'res2':  # TODO für resnets umschreiben
            # weight parameter corresponding to max_index
            if v1:
                best_layer_weight = freezed[2 + 3 * max_index]
            if v2:
                best_layer_weight = freezed[3*max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                # check only linear layers or activation functions
                if isinstance(child, torch.nn.Linear) or isinstance(child, kwargs_net['act_fun']):
                    # print(f'frist case type of child {child}')
                    new_model_children_list.append(child)
                    continue
                if not _is_freezed(child.l2.weight, freezed):
                    # print(f'second case type of child {child}')
                    new_model_children_list.append(child)
                    continue
                if v1 and child.l2.weight is best_layer_weight:
                    # print(f'third case type of child {child}')
                    # print(f'best layer weight is used')
                    child_for_return = child
                    new_model_children_list.append(child)
                    continue

                if v2 and child.l1.weight is best_layer_weight:  # for alternative param init
                    # print(f'third case type of child {child}')
                    # print(f'best layer weight is used')
                    child_for_return = child
                    new_model_children_list.append(child)
                    continue

            hidden_layers_new = len(new_model_children_list)-2
            new_kwargs_net = {'hidden_layers': hidden_layers_new,
                              'dim_hidden_layers': hidden_layers_new*[kwargs_net['dim_hidden_layers'][0]],
                              'act_fun': kwargs_net['act_fun'], 'type': _type}

            # for k in range(len(new_model_children_list)-1):  # build kwargs of the new net
            #    if isinstance(new_model_children_list[k], kwargs_net['act_fun']):
            #        new_kwargs_net['hidden_layers'] += 1
            #        new_kwargs_net['dim_hidden_layers'].append(
            #            new_model_children_list[k - 1].out_features)

        print(f'Insert layer at position {max_index} !')  # {2 * max_index} !')

    if mode == 'pos 0':
        # find index which has maximum absolute mean norm entry #
        max_index = 0

        if _type == 'fwd':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 * max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                # check only linear layers (no activation functions)
                if isinstance(child, torch.nn.Linear):
                    # if layer in model is not frozen or is the best layer,
                    # append the linear layer and Relu activation after
                    if not _is_freezed(child.weight, freezed):
                        new_model_children_list.append(child)
                        new_model_children_list.append(kwargs_net['act_fun']())

                    if child.weight is best_layer_weight:
                        child_for_return = child
                        new_model_children_list.append(child)
                        new_model_children_list.append(kwargs_net['act_fun']())

            new_kwargs_net = {'hidden_layers': 0, 'dim_hidden_layers': [
            ], 'act_fun': kwargs_net['act_fun'], 'type': _type}

            for k in range(len(new_model_children_list)-1):  # build kwargs of the new net
                if isinstance(new_model_children_list[k], kwargs_net['act_fun']):
                    new_kwargs_net['hidden_layers'] += 1
                    new_kwargs_net['dim_hidden_layers'].append(
                        new_model_children_list[k - 1].out_features)

        if _type == 'res2':  # TODO für resnets umschreiben
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 + 3 * max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                # check only linear layers or activation functions
                if isinstance(child, torch.nn.Linear) or isinstance(child, kwargs_net['act_fun']):
                    # print(f'frist case type of child {child}')
                    new_model_children_list.append(child)
                    continue
                if not _is_freezed(child.l2.weight, freezed):
                    # print(f'second case type of child {child}')
                    new_model_children_list.append(child)
                    continue
                if child.l2.weight is best_layer_weight:
                    # print(f'third case type of child {child}')
                    # print(f'best layer weight is used')
                    child_for_return = child
                    new_model_children_list.append(child)
                    continue

            hidden_layers_new = len(new_model_children_list)-2
            new_kwargs_net = {'hidden_layers': hidden_layers_new,
                              'dim_hidden_layers': hidden_layers_new*[kwargs_net['dim_hidden_layers'][0]],
                              'act_fun': kwargs_net['act_fun'], 'type': _type}

            # for k in range(len(new_model_children_list)-1):  # build kwargs of the new net
            #    if isinstance(new_model_children_list[k], kwargs_net['act_fun']):
            #        new_kwargs_net['hidden_layers'] += 1
            #        new_kwargs_net['dim_hidden_layers'].append(
            #            new_model_children_list[k - 1].out_features)

        print(f'Insert layer at position {max_index} !')  # {2 * max_index} !')

    if mode == 'abs min':

        # find index which has minimum mean norm entry
        min_index = min(range(len(freezed_norms_means_abs)),
                        key=lambda k: freezed_norms_means_abs[k])

        if _type == 'fwd':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 * min_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                # check only linear layers (no activation functions)
                if isinstance(child, torch.nn.Linear):
                    # if layer in model is not frozen or is the best layer,
                    # append the linear layer and Relu activation after
                    if not _is_freezed(child.weight, freezed):
                        new_model_children_list.append(child)
                        new_model_children_list.append(kwargs_net['act_fun']())

                    if child.weight is best_layer_weight:
                        child_for_return = child
                        new_model_children_list.append(child)
                        new_model_children_list.append(kwargs_net['act_fun']())

            new_kwargs_net = {'hidden_layers': 0, 'dim_hidden_layers': [
            ], 'act_fun': kwargs_net['act_fun'], 'type': _type}

            for k in range(len(new_model_children_list)-1):  # build kwargs of the new net
                if isinstance(new_model_children_list[k], kwargs_net['act_fun']):
                    new_kwargs_net['hidden_layers'] += 1
                    new_kwargs_net['dim_hidden_layers'].append(
                        new_model_children_list[k - 1].out_features)

        if _type == 'res2':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 + 3 * min_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                # check only linear layers or activation functions
                if isinstance(child, torch.nn.Linear) or isinstance(child, kwargs_net['act_fun']):
                    # print(f'frist case type of child {child}')
                    new_model_children_list.append(child)
                    continue
                if not _is_freezed(child.l2.weight, freezed):
                    # print(f'second case type of child {child}')
                    new_model_children_list.append(child)
                    continue
                if child.l2.weight is best_layer_weight:
                    # print(f'third case type of child {child}')
                    # print(f'best layer weight is used')
                    child_for_return = child
                    new_model_children_list.append(child)
                    continue

            hidden_layers_new = len(new_model_children_list)-2
            new_kwargs_net = {'hidden_layers': hidden_layers_new,
                              'dim_hidden_layers': hidden_layers_new*[kwargs_net['dim_hidden_layers'][0]],
                              'act_fun': kwargs_net['act_fun'], 'type': _type}

        print(f'Insert layer at position {min_index} !')  # {2 * max_index} !')

    if mode == 'max':
        # find index which has maximum mean norm entry
        max_index = max(range(len(freezed_norms_means)),
                        key=lambda l: freezed_norms_means[l])

        if _type == 'fwd':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 * max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                # check only linear layers (no activation functions)
                if isinstance(child, torch.nn.Linear):
                    # if layer in model is not frozen or is the best layer, append linear
                    # layer and Relu activation after
                    if not _is_freezed(child.weight, freezed) or child.weight is best_layer_weight:
                        new_model_children_list.append(child)
                        new_model_children_list.append(kwargs_net['act_fun']())

            new_kwargs_net = {'hidden_layers': 0, 'dim_hidden_layers': [
            ], 'act_fun': kwargs_net['act_fun'], 'type': _type}

            for k in range(len(new_model_children_list)-1):  # build kwargs of the new net
                if isinstance(new_model_children_list[k], kwargs_net['act_fun']):
                    new_kwargs_net['hidden_layers'] += 1
                    new_kwargs_net['dim_hidden_layers'].append(
                        new_model_children_list[k - 1].out_features)

        if _type == 'res2':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 + 3 * max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                # check only linear layers or activation functions
                if isinstance(child, torch.nn.Linear) or isinstance(child, kwargs_net['act_fun']):
                    # print(f'frist case type of child {child}')
                    new_model_children_list.append(child)
                    continue
                if not _is_freezed(child.l2.weight, freezed):
                    # print(f'second case type of child {child}')
                    new_model_children_list.append(child)
                    continue
                if child.l2.weight is best_layer_weight:
                    # print(f'third case type of child {child}')
                    # print(f'best layer weight is used')
                    new_model_children_list.append(child)
                    continue

            hidden_layers_new = len(new_model_children_list)-2
            new_kwargs_net = {'hidden_layers': hidden_layers_new,
                              'dim_hidden_layers': hidden_layers_new*[kwargs_net['dim_hidden_layers'][0]],
                              'act_fun': kwargs_net['act_fun'], 'type': _type}

        print(f'Insert layer at position {max_index} !')  # {2 * max_index} !')

    if mode == 'min':
        # find index which has minimum mean norm entry
        min_index = min(range(len(freezed_norms_means)),
                        key=lambda k: freezed_norms_means[k])

        if _type == 'fwd':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 * min_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                # check only linear layers (no activation functions)
                if isinstance(child, torch.nn.Linear):
                    # if layer in model is not frozen or is the best layer, append
                    # the linear layer
                    # and Relu activation after
                    if not _is_freezed(child.weight, freezed) or child.weight is best_layer_weight:
                        new_model_children_list.append(child)
                        new_model_children_list.append(kwargs_net['act_fun']())

            new_kwargs_net = {'hidden_layers': 0, 'dim_hidden_layers': [
            ], 'act_fun': kwargs_net['act_fun'], 'type': _type}

            for k in range(len(new_model_children_list)-1):  # build kwargs of the new net
                if isinstance(new_model_children_list[k], kwargs_net['act_fun']):
                    new_kwargs_net['hidden_layers'] += 1
                    new_kwargs_net['dim_hidden_layers'].append(
                        new_model_children_list[k - 1].out_features)

        if _type == 'res2':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 + 3 * min_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                # check only linear layers or activation functions
                if isinstance(child, torch.nn.Linear) or isinstance(child, kwargs_net['act_fun']):
                    # print(f'frist case type of child {child}')
                    new_model_children_list.append(child)
                    continue
                if not _is_freezed(child.l2.weight, freezed):
                    # print(f'second case type of child {child}')
                    new_model_children_list.append(child)
                    continue
                if child.l2.weight is best_layer_weight:
                    # print(f'third case type of child {child}')
                    # print(f'best layer weight is used')
                    new_model_children_list.append(child)
                    continue

            hidden_layers_new = len(new_model_children_list)-2
            new_kwargs_net = {'hidden_layers': hidden_layers_new,
                              'dim_hidden_layers': hidden_layers_new*[kwargs_net['dim_hidden_layers'][0]],
                              'act_fun': kwargs_net['act_fun'], 'type': _type}

        print(f'Insert layer at position {min_index} !')  # {2 * max_index} !')

    if _type == 'fwd':
        return torch.nn.Sequential(*new_model_children_list[:-1]), new_kwargs_net, child_for_return
    if _type == 'res2':
        return torch.nn.Sequential(*new_model_children_list), new_kwargs_net, child_for_return
