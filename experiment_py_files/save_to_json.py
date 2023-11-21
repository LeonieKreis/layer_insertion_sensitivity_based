import json


def write_losses(path, losses, max_length, structures=None, errors=None, interval_testerror=None, times=None, grad_norms = None, its_per_epoch=1):
    '''
    saves losses in json file under 'path' in a dict,
    optionally saves also the information which loss happened on which model
    when structures is given. In order to have the same length, for the std and mean function,
    nan values get appended to the end of losses until the list has the length max_length.
    also errors are saved, the intervals must be specified with interval testerror and its_per_epoch. The remaining iterations are filled with nan.
    further, the exit flag can be saved
    '''

    try:
        with open(path) as file:
            data = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        data = {}

    if len(losses) < max_length:
        diff = max_length - len(losses)
        losses += diff*[float("nan")]

    # save errors at the right points
    errs = (max_length+1)*[float("nan")]
    ind = [i * interval_testerror * its_per_epoch for i in range(len(errs))]
    for i, e in zip(ind, errors):
        errs[i] = e

    number = len(data.keys())

    if grad_norms is None:
        if times is None:
            if structures is None and errors is None:
                data[str(number)] = {'losses': losses}
            if structures is not None and errors is None:
                data[str(number)] = {'losses': losses,
                                    'structures': structures}
            if structures is None and errors is not None:
                data[str(number)] = {'losses': losses,
                                    'errors': errs}
            if structures is not None and errors is not None:
                data[str(number)] = {'losses': losses,
                                    'structures': structures,
                                    'errors': errs}

        if times is not None:
            if structures is None and errors is None:
                data[str(number)] = {'losses': losses,
                                    'times': times}
            if structures is not None and errors is None:
                data[str(number)] = {'losses': losses,
                                    'structures': structures,
                                    'times': times}
            if structures is None and errors is not None:
                data[str(number)] = {'losses': losses,
                                    'errors': errs,
                                    'times': times}
            if structures is not None and errors is not None:
                data[str(number)] = {'losses': losses,
                                    'structures': structures,
                                    'errors': errs,
                                    'times': times}
                
    if grad_norms is not None:
        if times is None:
            if structures is None and errors is None:
                data[str(number)] = {'losses': losses, 
                                     'grad_norms': grad_norms}
            if structures is not None and errors is None:
                data[str(number)] = {'losses': losses,
                                    'structures': structures, 
                                    'grad_norms': grad_norms}
            if structures is None and errors is not None:
                data[str(number)] = {'losses': losses,
                                    'errors': errs, 
                                    'grad_norms': grad_norms}
            if structures is not None and errors is not None:
                data[str(number)] = {'losses': losses,
                                    'structures': structures,
                                    'errors': errs, 
                                    'grad_norms': grad_norms}

        if times is not None:
            if structures is None and errors is None:
                data[str(number)] = {'losses': losses,
                                    'times': times, 
                                    'grad_norms': grad_norms}
            if structures is not None and errors is None:
                data[str(number)] = {'losses': losses,
                                    'structures': structures,
                                    'times': times, 
                                    'grad_norms': grad_norms}
            if structures is None and errors is not None:
                data[str(number)] = {'losses': losses,
                                    'errors': errs,
                                    'times': times, 
                                    'grad_norms': grad_norms}
            if structures is not None and errors is not None:
                data[str(number)] = {'losses': losses,
                                    'structures': structures,
                                    'errors': errs,
                                    'times': times, 
                                    'grad_norms': grad_norms}


    with open(path, 'w') as file:
        json.dump(data, file)
