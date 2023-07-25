import torch


@torch.no_grad()
def simple_backtracking(
        model: torch.nn.Sequential,
        initial_step_size:  float,
        diminishing_factor:  float,
        curr_loss: torch.tensor,
        x,
        y,
        max_it=100):
    """Performs simple backtracking line search until descent is observed or max_it reached

    Args:
        model (torch.nn.Sequential): the current model
        initial_step_size (torch.tensor | float): initial step size
        diminishing_factor (torch.tensor | float): step size, by which the lr is reduced in each iteration
        curr_loss (torch.tensor): curr loss value, the value, that has to be reached to be a succesfull step
        x: Input for model (the current batch)
        y: Desired output (the categories of the current batch)
        max_it (int, optional): maximum number of iterations. Defaults to 100.

    Returns:
       the lr, which has been used in performing a succesfull step. The step is not performed
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = initial_step_size
    # print(f'lr before backtr as input {lr}')

    for it in range(max_it):
        _apply_grad_step(model, lr)
        new_loss = loss_fn(model(x), y)
        if new_loss < curr_loss:
            print(f'successful backtracking iteration: {it}')
            print(f'resulting learning rate: {lr}')
            _undo_grad_step(model, lr)
            return lr
        _undo_grad_step(model, lr)
        lr *= diminishing_factor
    print(' no lr found!')
    return lr


def _apply_grad_step(model: torch.nn.Sequential, lr):
    """Helper function, performs a gradient step

    Args:
        model (torch.nn.Sequential): the model
        lr: The learing rate to be used
    """
    for p in model.parameters():
        p.add_(p.grad, alpha=- lr)


def _undo_grad_step(model: torch.nn.Sequential, lr):
    """Helper function, performs a negative gradient step

    Args:
        model (torch.nn.Sequential): the model
        lr: The learing rate to be used
    """
    return _apply_grad_step(model, - lr)
