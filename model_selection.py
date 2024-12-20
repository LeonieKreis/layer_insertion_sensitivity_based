import torch
import nets
from model_utils import freeze_params, _is_freezed


def tmp_net(dim_in, dim_out, hidden_layers, dim_hidden_layers, act_fun, _type, model, v2=False):
    '''
    builds a new model for the equality constrained training with frozen parameters based on an old/coarse model.

    Args:
        dim_in (int): (flattened) dimesion of the images
        dim_out (int): number of classes
        hidden_layers (int): number of hidden layers in the old/coarse model
        dim_hidden_layers (int or list): the width of the hidden layers. If int, all hidden layers have the same width.
        act_fun: activation function of the model
        _type: type of the model
        model: old/coarse pytorch model
        v2 (default False): determines for 2weight resnets whether the init strategy with zeros in the outer or inner weight is taken.
        v2 corresponds to the more restrictive version where the inner weight is initialized with a zero matrix.

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

        new_model = nets.feed_forward(
            dim_in, dim_out, hidden_layers_new, dim_hidden_layers_new, act_fun=act_fun)  # build new model
        # freeze and initialize every second layer of the new model
        freezed = freeze_params(new_model, act_fun, model, _type='fwd')

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
            hidden_layers_new = 2 * hidden_layers # - 1  # TODO
            dim_hidden_layers_new = hidden_layers_new * \
                [dim_hidden_layers[0]]

        new_kwargs_net = {'hidden_layers': hidden_layers_new,
                          'dim_hidden_layers': dim_hidden_layers_new,
                          'act_fun': act_fun,
                          'type': _type}

        new_model = nets.two_weight_resnet(dim_in, dim_out,
                                           hidden_layers_new, dim_hidden_layers_new, act_fun=act_fun)  # build new model
        # freeze and initialize every second layer of the new model
        freezed = freeze_params(new_model, act_fun, model, _type='res2', v2=v2)

        # the not-frozen layers get initialized with the parameter values from the old/coarse model
        with torch.no_grad():
            old_param_iterator = model.parameters()
            for p_new in new_model.parameters():
                if not _is_freezed(p_new, freezed):
                    p = next(old_param_iterator)
                    p_new.copy_(p)              
        return new_model, freezed, new_kwargs_net
    
    if _type == 'res1':

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

        new_model = nets.one_weight_resnet(dim_in, dim_out,
                                           hidden_layers_new, dim_hidden_layers_new, act_fun=act_fun)  # build new model
        # freeze and initialize every second layer of the new model
        freezed = freeze_params(new_model, act_fun, model, _type='res1', v2=v2)

        # the not-frozen layers get initialized with the parameter values from the old/coarse model
        with torch.no_grad():
            old_param_iterator = model.parameters()
            for p_new in new_model.parameters():
                if not _is_freezed(p_new, freezed):
                    p = next(old_param_iterator)
                    p_new.copy_(p)
        return new_model, freezed, new_kwargs_net



def select_new_model(avg_grad_norm, freezed_norms, model, freezed, kwargs_net, mode='abs max', _type='fwd', v2=False):
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
        kwargs_net: dict, which contains the info about model
        mode (string): either 'min' or 'max', 'abs min' or 'abs max' so far, indicating whether the layer with th4
        largest or smallest or absolute value smallest (comparison) or absolute value largest (theory)
        lagrange multipliers are chosen in model selection. default is 'abs max'.
        _type: type of the model
        v2 (default False): determines for 2weight resnets whether the init strategy with zeros in the outer or inner weight is taken.
        v2 corresponds to the more restrictive version where the inner weight is initialized with a zero matrix.


    Out:
        new_model: model which has the selected new layer initilazied with identity and zero but not frozen.
        new_kwargs_net (dict): kwargs of new_model
        child_for_return: child of the new net which is the newly inserted one (ccan be used for training only the new parameters)
    '''
    # look only at relevant weights for the sensitivities ##############################################################
    freezed_norms_only_relevant_weights = []

    if _type == 'fwd':
        for k, freezed_norm in enumerate(freezed_norms):
            if k % 2 == 0:
                freezed_norms_only_relevant_weights.append(freezed_norm)

    if _type == 'res1':
        for k, freezed_norm in enumerate(freezed_norms):
            if k % 2 == 0:
                freezed_norms_only_relevant_weights.append(freezed_norm)

    if _type == 'res2':
        for k, freezed_norm in enumerate(freezed_norms):
            v1 = not v2
            if v1:
                weight = 2
            if v2:
                weight = 0
            if k % 3 == weight:
                freezed_norms_only_relevant_weights.append(freezed_norm)

    print(f'the averaged shadow prices  of all available positions: {freezed_norms_only_relevant_weights}')
    
    sens = [ob.item() for ob in freezed_norms_only_relevant_weights]

    # select layer based on different criteria ########################################################################
    if mode == 'abs max':
        # find index which has maximum absolute mean norm entry #
        max_index = max(range(len(freezed_norms_only_relevant_weights)),
                        key=lambda l: freezed_norms_only_relevant_weights[l])

        if _type == 'fwd':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 * max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                if isinstance(child, torch.nn.Flatten):  # handle flatten at beginning
                    new_model_children_list.append(child)
                    continue
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
                    
        if _type == 'res1':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 * max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                if isinstance(child, torch.nn.Flatten):  # handle flatten at beginning
                    new_model_children_list.append(child)
                    continue
                # check only linear layers (no activation functions)
                if isinstance(child, torch.nn.Linear):
                    new_model_children_list.append(child)
                    continue
                if isinstance(child, kwargs_net['act_fun']):
                    new_model_children_list.append(child)
                    continue
                if not _is_freezed(child.l1.weight, freezed):
                    new_model_children_list.append(child)
                    continue
                if child.l1.weight is best_layer_weight:
                    child_for_return=child
                    new_model_children_list.append(child)
                    continue
            # minus one linear and actfun and flatten
            hidden_layers_new = len(new_model_children_list)-3
            new_kwargs_net = {'hidden_layers': hidden_layers_new,
                              'dim_hidden_layers': hidden_layers_new*[kwargs_net['dim_hidden_layers'][0]],
                              'act_fun': kwargs_net['act_fun'], 'type': _type}
            
        




        if _type == 'res2':
            # weight parameter corresponding to max_index
            if v1:
                best_layer_weight = freezed[2 + 3 * max_index]
            if v2:
                best_layer_weight = freezed[3*max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                if isinstance(child, torch.nn.Flatten):  # handle flatten at beginning
                    new_model_children_list.append(child)
                    continue
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

            # minus one linear and actfun and flatten
            hidden_layers_new = len(new_model_children_list)-3
            new_kwargs_net = {'hidden_layers': hidden_layers_new,
                              'dim_hidden_layers': hidden_layers_new*[kwargs_net['dim_hidden_layers'][0]],
                              'act_fun': kwargs_net['act_fun'], 'type': _type}

        print(f'Insert layer at position {max_index} !')

    if mode == 'pos 0':
        # find index which has maximum absolute mean norm entry #
        max_index = 0

        if _type == 'res1':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 * max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                if isinstance(child, torch.nn.Flatten):  # handle flatten at beginning
                    new_model_children_list.append(child)
                    continue
                # check only linear layers (no activation functions)
                if isinstance(child, torch.nn.Linear):
                    new_model_children_list.append(child)
                    continue
                if isinstance(child, kwargs_net['act_fun']):
                    new_model_children_list.append(child)
                    continue
                if not _is_freezed(child.l1.weight, freezed):
                    new_model_children_list.append(child)
                    continue
                if child.l1.weight is best_layer_weight:
                    child_for_return=child
                    new_model_children_list.append(child)
                    continue
            # minus one linear and actfun and flatten
            hidden_layers_new = len(new_model_children_list)-3
            new_kwargs_net = {'hidden_layers': hidden_layers_new,
                              'dim_hidden_layers': hidden_layers_new*[kwargs_net['dim_hidden_layers'][0]],
                              'act_fun': kwargs_net['act_fun'], 'type': _type}

        if _type == 'fwd':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 * max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                if isinstance(child, torch.nn.Flatten):  # handle flatten at beginning
                    new_model_children_list.append(child)
                    continue
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

        if _type == 'res2':  # todo für v1 v2 handlen!
            # weight parameter corresponding to max_index
            if v1:
                best_layer_weight = freezed[2 + 3 * max_index]
            if v2:
                best_layer_weight = freezed[3*max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                if isinstance(child, torch.nn.Flatten):  # handle flatten at beginning
                    new_model_children_list.append(child)
                    continue
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

            hidden_layers_new = len(new_model_children_list)-3
            # minus one linear and actfun and flatten
            new_kwargs_net = {'hidden_layers': hidden_layers_new,
                              'dim_hidden_layers': hidden_layers_new*[kwargs_net['dim_hidden_layers'][0]],
                              'act_fun': kwargs_net['act_fun'], 'type': _type}

        print(f'Insert layer at position {max_index} !')

    if mode == 'abs min':

        # find index which has minimum mean norm entry
        min_index = min(range(len(freezed_norms_only_relevant_weights)),
                        key=lambda k: freezed_norms_only_relevant_weights[k])
        
        if _type == 'res1':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 * min_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                if isinstance(child, torch.nn.Flatten):  # handle flatten at beginning
                    new_model_children_list.append(child)
                    continue
                # check only linear layers (no activation functions)
                if isinstance(child, torch.nn.Linear):
                    new_model_children_list.append(child)
                    continue
                if isinstance(child, kwargs_net['act_fun']):
                    new_model_children_list.append(child)
                    continue
                if not _is_freezed(child.l1.weight, freezed):
                    new_model_children_list.append(child)
                    continue
                if child.l1.weight is best_layer_weight:
                    child_for_return=child
                    new_model_children_list.append(child)
                    continue
            # minus one linear and actfun and flatten
            hidden_layers_new = len(new_model_children_list)-3
            new_kwargs_net = {'hidden_layers': hidden_layers_new,
                              'dim_hidden_layers': hidden_layers_new*[kwargs_net['dim_hidden_layers'][0]],
                              'act_fun': kwargs_net['act_fun'], 'type': _type}

        if _type == 'fwd':
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 * min_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                if isinstance(child, torch.nn.Flatten):  # handle flatten at beginning
                    new_model_children_list.append(child)
                    continue
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
            if v1:
                best_layer_weight = freezed[2 + 3 * min_index]
            if v2:
                best_layer_weight = freezed[3*min_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                if isinstance(child, torch.nn.Flatten):  # handle flatten at beginning
                    new_model_children_list.append(child)
                    continue
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

            hidden_layers_new = len(new_model_children_list)-3
            new_kwargs_net = {'hidden_layers': hidden_layers_new,
                              'dim_hidden_layers': hidden_layers_new*[kwargs_net['dim_hidden_layers'][0]],
                              'act_fun': kwargs_net['act_fun'], 'type': _type}

        print(f'Insert layer at position {min_index} !')

    if mode == 'threshold':
        # average the gradients of the unfrozen parameters
        avg_grad_norm_only_weights = []
        if _type == 'fwd':
            for k, avg_norm in enumerate(avg_grad_norm):
                if k % 2 == 0:
                    avg_grad_norm_only_weights.append(avg_norm)

        if _type == 'res1':
            for k, avg_norm in enumerate(avg_grad_norm):
                if k % 2 == 0:
                    avg_grad_norm_only_weights.append(avg_norm)

        if _type == 'res2':
            for k, avg_norm in enumerate(avg_grad_norm):
                v1 = not v2
                if v1:
                    weight = 2
                if v2:
                    weight = 0
                if k % 3 == weight:
                    avg_grad_norm_only_weights.append(avg_norm)


        avg = torch.mean(torch.tensor(avg_grad_norm_only_weights))
        tau = 1.
        good_new_layers = [
            x > tau*avg for x in freezed_norms_only_relevant_weights]

        max_indices = []
        for i in good_new_layers:
            if i is True:
                max_indices.append(i)

        if len(max_indices) == 0:
            # no new layer is inserted
            best_layer_weight = None
            max_index = None

        if len(max_indices)==1:
            max_index = max_indices[0]

        if len(max_indices) > 1:  # TODO: handle insertion of multiple layers
            # for now we only choose the first one
            max_index = max_indices[0]

        if _type == 'fwd':
            child_for_return = 0
            # weight parameter corresponding to max_index
            if best_layer_weight is not None:
                best_layer_weight = freezed[2 * max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                if isinstance(child, torch.nn.Flatten):  # handle flatten at beginning
                    new_model_children_list.append(child)
                    continue
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

        if _type == 'res1':
            child_for_return = 0
            # weight parameter corresponding to max_index
            best_layer_weight = freezed[2 * max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                if isinstance(child, torch.nn.Flatten):  # handle flatten at beginning
                    new_model_children_list.append(child)
                    continue
                # check only linear layers (no activation functions)
                if isinstance(child, torch.nn.Linear):
                    new_model_children_list.append(child)
                    continue
                if isinstance(child, kwargs_net['act_fun']):
                    new_model_children_list.append(child)
                    continue
                if not _is_freezed(child.l1.weight, freezed):
                    new_model_children_list.append(child)
                    continue
                if child.l1.weight is best_layer_weight:
                    child_for_return=child
                    new_model_children_list.append(child)
                    continue
            # minus one linear and actfun and flatten
            hidden_layers_new = len(new_model_children_list)-3
            new_kwargs_net = {'hidden_layers': hidden_layers_new,
                              'dim_hidden_layers': hidden_layers_new*[kwargs_net['dim_hidden_layers'][0]],
                              'act_fun': kwargs_net['act_fun'], 'type': _type}

        if _type == 'res2':  # todo für v1 v2 handlen!
            child_for_return=0
            # weight parameter corresponding to max_index
            if best_layer_weight is not None:
                if v1:
                    best_layer_weight = freezed[2 + 3 * max_index]
                if v2:
                    best_layer_weight = freezed[3 * max_index]

            new_model_children_list = []  # list for storing all layers for the new model

            for child in model.children():  # iterate over all parameters of the eq-constr model
                if isinstance(child, torch.nn.Flatten):  # handle flatten at beginning
                    new_model_children_list.append(child)
                    continue
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

            hidden_layers_new = len(new_model_children_list)-3
            # minus one linear and actfun and flatten
            new_kwargs_net = {'hidden_layers': hidden_layers_new,
                              'dim_hidden_layers': hidden_layers_new*[kwargs_net['dim_hidden_layers'][0]],
                              'act_fun': kwargs_net['act_fun'], 'type': _type}

        print(f'Insert layer at position {max_index} !')

    if _type == 'fwd':
        return torch.nn.Sequential(*new_model_children_list[:-1]), new_kwargs_net, child_for_return, sens
    if _type == 'res2':
        return torch.nn.Sequential(*new_model_children_list), new_kwargs_net, child_for_return, sens
    if _type == 'res1':
        return torch.nn.Sequential(*new_model_children_list), new_kwargs_net, child_for_return, sens