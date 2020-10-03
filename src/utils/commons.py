from statistics import mean


def best_fit_slope_and_intercept(x_values, y_values):
    mean_xs, mean_ys = mean(x_values), mean(y_values)
    mean_xs_ys = mean(x_values * y_values)
    m = ((mean_xs * mean_ys) - mean_xs_ys) / ((mean_xs ** 2) - mean(x_values ** 2))
    b = mean_ys - m * mean_xs
    return m, b
