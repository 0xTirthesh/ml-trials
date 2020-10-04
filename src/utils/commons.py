import logging
import logging.config
import random
from statistics import mean
from typing import Optional

import numpy as np

from settings import CONFIG_DIR


def init_logging():
    logging.config.fileConfig(f'{CONFIG_DIR}/logging.conf')
    return


def best_fit_slope_and_intercept(x_values, y_values):
    mean_xs, mean_ys = mean(x_values), mean(y_values)
    mean_xs_ys = mean(x_values * y_values)
    m = ((mean_xs * mean_ys) - mean_xs_ys) / ((mean_xs ** 2) - mean(x_values ** 2))
    b = mean_ys - m * mean_xs
    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = np.array([mean(ys_orig) for y in ys_orig], dtype = np.float64)
    squared_error_regression = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regression / squared_error_y_mean)


def create_dataset(no_of_data_points, variance, step = 2, correlation: Optional[str] = None):
    val = 1
    ys = []
    for i in range(no_of_data_points):
        y = val + random.randrange(-variance, variance)
        ys.append(y)

        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)
