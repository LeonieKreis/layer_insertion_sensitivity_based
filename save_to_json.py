import json


def write_losses(path, losses, max_length, structures=None, errors=None, interval_testerror=None, 
                 times=None, grad_norms=None, final_params=None,
                 exit_flag=None, its_per_epoch=1,sens=None):
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

    # if exit_flag is None:
    #     if structures is None and errors is None:
    #         data[str(number)] = {'losses': losses}
    #     if structures is not None and errors is None:
    #         data[str(number)] = {'losses': losses,
    #                             'structures': structures}
    #     if structures is None and errors is not None:
    #         data[str(number)] = {'losses': losses,
    #                             'errors': errs}
    #     if structures is not None and errors is not None:
    #         data[str(number)] = {'losses': losses,
    #                             'structures': structures,
    #                             'errors': errs}

    # if exit_flag is not None:
    #     if structures is None and errors is None:
    #         data[str(number)] = {'losses': losses,
    #                              'exit_flag': exit_flag}
    #     if structures is not None and errors is None:
    #         data[str(number)] = {'losses': losses,
    #                             'structures': structures,
    #                              'exit_flag': exit_flag}
    #     if structures is None and errors is not None:
    #         data[str(number)] = {'losses': losses,
    #                             'errors': errs,
    #                              'exit_flag': exit_flag}
    #     if structures is not None and errors is not None:
    #         data[str(number)] = {'losses': losses,
    #                             'structures': structures,
    #                             'errors': errs,
    #                              'exit_flag': exit_flag}
            
    # data[str(number)]= {'losses': losses,
    #                     'structures': structures,
    #                     'errors': errs,
    #                     'times': times,
    #                     'grad_norms': grad_norms,
    #                     'exit_flag': exit_flag,
    #                     'final_params': final_params,
    #                     'sens': sens}
    

    number_dict = {'losses': losses}
    if structures is not None:
        number_dict['structures'] = structures
    if errors is not None:
        number_dict['errors'] = errs
    if times is not None:
        number_dict['times'] = times
    if grad_norms is not None:
        number_dict['grad_norms'] = grad_norms
    if exit_flag is not None:
        number_dict['exit_flag'] = exit_flag
    if final_params is not None:
        number_dict['final_params'] = final_params
    if sens is not None:
        number_dict['sens'] = sens

    data[str(number)] = number_dict

    
    with open(path, 'w') as file:
        json.dump(data, file)

