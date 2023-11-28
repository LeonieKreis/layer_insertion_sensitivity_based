from torch import nn


def feed_forward(dim_in, dim_out, hidden_layers=1, dim_hidden_layers=3, act_fun=nn.ReLU, type='fwd', flatten = True):
    '''
    Generates feedforward model.

    Args:
        hidden_layers (int): default 1, specifies the number of hidden layers in the model.
        dim_hidden_layers (int or list): specifies the widths of the hidden layers.
        act_fun: default nn.ReLU, but any (for layer insertion suitable) activation function, implemented such that it
        can be used in sequential model construction (must be wrapped as a module).
        flatten: (bool) indicates whether data is an image and must be flattened or not (as it is the case for the spirals)

    Out:
        sequential pytorch model. If input is invalid, raise ValueError.
    '''
    modules = []
    if hidden_layers == 0:  # if no hidden layers are specified, the net is a linear model
        if flatten:
            return nn.Sequential(nn.Flatten(), nn.Linear(dim_in, dim_out))
        else: return nn.Sequential(nn.Linear(dim_in, dim_out))
    if isinstance(dim_hidden_layers, int):  # all hidden layers have the same width
        dim_hidden_layers = [dim_hidden_layers] * hidden_layers
    # the hidden layers have different widths
    if isinstance(dim_hidden_layers, (list, tuple)):
        dim_old = dim_in
        if flatten:
            modules.append(nn.Flatten())
        for dim in dim_hidden_layers:
            dim_curr = dim
            modules.append(
                nn.Linear(in_features=dim_old, out_features=dim_curr))
            modules.append(act_fun())
            dim_old = dim_curr

        modules.append(nn.Linear(dim_old, dim_out))

        return nn.Sequential(*modules)

    raise ValueError()


# ResBlock1 is a residual block with two weights and one bias, which we consider mainly
class ResBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, h=1, act_fun=nn.ReLU):
        super().__init__()
        self.af1 = act_fun()
        self.l1 = nn.Linear(in_channels, out_channels)
        self.shortcut = nn.Sequential()
        self.l2 = nn.Linear(out_channels, out_channels, bias=False)
        self.h = h

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = self.l1(input)
        input = self.af1(input)
        input = self.l2(input)
        input = self.h*input + shortcut
        return input


def two_weight_resnet(dim_in, dim_out, hidden_layers=1, dim_hidden_layers=3, act_fun=nn.ReLU, type='res2', flatten=True):
    '''
    Generates resnet with two weights- model.

    Args:
        hidden_layers (int): default 1, specifies the number of hidden layers in the model.
        dim_hidden_layers (int or list): specifies the widths of the hidden layers.
        act_fun: default nn.ReLU, but any (for layer insertion suitable) activation function, implemented such that it
        can be used in sequential model construction (must be wrapped as a module).
        flatten: (bool) indicates whether data is an image and must be flattened or not (as it is the case for the spirals)

    Out:
        sequential pytorch model. If input is invalid, raise ValueError.
    '''
    modules = []
    if hidden_layers == 0:  # if no hidden layers are specified, the net is a linear model
        if flatten:
            return nn.Sequential(nn.Flatten(), nn.Linear(dim_in, dim_out))
        else:
            return nn.Sequential(nn.Linear(dim_in, dim_out))
    if hidden_layers == 1:
        if flatten:
            return (nn.Sequential( nn.Linear(dim_in, dim_hidden_layers), act_fun(),
                              nn.Linear(dim_hidden_layers, dim_out)))
        else:
            return (nn.Sequential(nn.Flatten(), nn.Linear(dim_in, dim_hidden_layers), act_fun(),
                              nn.Linear(dim_hidden_layers, dim_out)))
    if isinstance(dim_hidden_layers, int):  # all hidden layers have the same width
        dim_hidden_layers = [dim_hidden_layers] * hidden_layers
    # the hidden layers have the same widths
    if isinstance(dim_hidden_layers, (list, tuple)):
        dim_old = dim_in
        if flatten:
            modules.append(nn.Flatten())
        for k, dim in enumerate(dim_hidden_layers):
            dim_curr = dim
            if k == 0:
                modules.append(
                    nn.Linear(in_features=dim_old, out_features=dim_curr))
                modules.append(act_fun())

            else:
                modules.append(
                    ResBlock1(dim_old, dim_curr, h=1., act_fun=act_fun))

            dim_old = dim_curr

        modules.append(nn.Linear(dim_old, dim_out))

        return nn.Sequential(*modules)

    raise ValueError()


# ResBlock is a residual block with one weight and one bias
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,h=1., act_fun=nn.ReLU):
        super().__init__()
        self.af1 = act_fun()
        self.l1 = nn.Linear(in_channels, out_channels)
        self.shortcut = nn.Sequential()
        self.h = h

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = self.l1(input)
        input = self.af1(input)
        input = self.h * input + shortcut
        return input


def one_weight_resnet(dim_in, dim_out, hidden_layers=1, dim_hidden_layers=3, act_fun=nn.ReLU, type='res1'):
    '''
    Generates resnet with one-weight- model.

    Args:
        hidden_layers (int): default 1, specifies the number of hidden layers in the model.
        dim_hidden_layers (int or list): specifies the widths of the hidden layers.
        act_fun: default nn.ReLU, but any (for layer insertion suitable) activation function, implemented such that it
        can be used in sequential model construction (must be wrapped as a module).

    Out:
        sequential pytorch model. If input is invalid, raise ValueError.
    '''
    modules = []
    if hidden_layers == 0:  # if no hidden layers are specified, the net is a linear model
        return nn.Sequential(nn.Flatten(), nn.Linear(dim_in, dim_out))
    if hidden_layers == 1:
        return (nn.Sequential(nn.Flatten(), nn.Linear(dim_in, dim_hidden_layers), act_fun(),
                              nn.Linear(dim_hidden_layers, dim_out)))
    if isinstance(dim_hidden_layers, int):  # all hidden layers have the same width
        dim_hidden_layers = [dim_hidden_layers] * hidden_layers
    # the hidden layers have the same widths
    if isinstance(dim_hidden_layers, (list, tuple)):
        dim_old = dim_in
        modules.append(nn.Flatten())
        for k, dim in enumerate(dim_hidden_layers):
            dim_curr = dim
            if k == 0:
                modules.append(
                    nn.Linear(in_features=dim_old, out_features=dim_curr))
                modules.append(act_fun())

            else:
                modules.append(
                    ResBlock(dim_old, dim_curr, h=1., act_fun=act_fun))

            dim_old = dim_curr

        modules.append(nn.Linear(dim_old, dim_out))

        return nn.Sequential(*modules)

    raise ValueError()
