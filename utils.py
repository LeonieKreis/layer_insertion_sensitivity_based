from typing import List
import numpy as np
import datetime

def get_timestamp():
    current_time = datetime.datetime.now()
    timestamp = (
        str(current_time.year)
        + "-"
        + str(current_time.month)
        + "-"
        + str(current_time.day)
        + "_"
        + str(current_time.hour)
        + "-"
        + str(current_time.minute)
        + "-"
        + str(current_time.second)
        + "-"
        + str(current_time.microsecond)
    )
    return timestamp


def ema(data: List, gamma=.9):
    '''
    Implemnets exponential moving average.

    Args:
        data (list): list of data
        gamma (float): decay factor. default 0.9

    Out:
        list of em-averaged data.
    '''
    res = []
    curr = data[0]

    res.append(float(curr))
    for val in data[1:]:
        curr = gamma * curr + (1 - gamma) * val  # EMA
        res.append(float(curr))
    return res


def ema_np(data, gamma=0.9):
    '''
    Implemnets exponential moving average.

    Args:
        data (list): list of data
        gamma (float): decay factor. default 0.9

    Out:
        list of em-averaged data.
    '''
    l_ema = ema(list(data), gamma=gamma)
    res = np.array(l_ema)
    return res


def ema2(data: List, gamma=.9, no_of_steps_back = 10):
    '''
    Implements average over a fixed window of 'no_of_steps_back' iterations
    '''
    res = []
    curr = data[0]
    
    i=0
    res.append(float(curr))
    for val in data[1:]:
        min_index = int(max(0,i-no_of_steps_back))
        curr = gamma* sum(data[min_index:i+1])/len(data[min_index:i+1]) + (1-gamma)* val
        res.append(float(curr))
        i+=1
    return res

def ema2_np(data, gamma=0.9, no_of_steps_back = 10):
    
    l_ema = ema2(list(data), gamma=gamma, no_of_steps_back=no_of_steps_back)
    res = np.array(l_ema)
    return res

