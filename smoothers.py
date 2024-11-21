def moving_average(data, window_size):
    """
    Applies a moving average smoothing to a given dataset.

    Parameters:
    - data: List or NumPy array containing the input data.
    - window_size: Size of the moving average window.

    Returns:
    - smoothed_data: List containing the smoothed data.
    """
    smoothed_data = []

    for i in range(len(data)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(data), i + window_size // 2 + 1)

        # Calculate the average within the window
        average_value = sum(data[start_index:end_index]) / (end_index - start_index)

        smoothed_data.append(average_value)

    return smoothed_data
