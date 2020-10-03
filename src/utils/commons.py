from statistics import mean

import numpy as np


def best_fit_slope_and_intercept(x_values, y_values):
    mean_xs, mean_ys = mean(x_values), mean(y_values)
    mean_xs_ys = mean(x_values * y_values)
    m = ((mean_xs * mean_ys) - mean_xs_ys) / ((mean_xs ** 2) - mean(x_values ** 2))
    b = mean_ys - m * mean_xs
    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = np.array([mean(ys_orig) for y in ys_orig])
    squared_error_regression = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regression / squared_error_y_mean)
