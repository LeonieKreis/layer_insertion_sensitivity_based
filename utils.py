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
