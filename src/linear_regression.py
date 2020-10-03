import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

from utils.commons import best_fit_slope_and_intercept, coefficient_of_determination

style.use('fivethirtyeight')

PLOT_XY = False
PLOT_XY_PREDICT = True

xs = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)
ys = np.array([7, 12, 16, 19, 21, 25], dtype = np.float64)

if PLOT_XY:
    plt.scatter(xs, ys)
    plt.show()

m, b = best_fit_slope_and_intercept(xs, ys)

print("Slope: ", m)
print("Intercept: ", b)

# model: y = mx + b
regression_line = [(m * x) + b for x in xs]

predict_x = 8
predict_y = m * predict_x + b

r_squared = coefficient_of_determination(ys, regression_line)

print("R-squared: ", r_squared)

if PLOT_XY_PREDICT:
    plt.scatter(xs, ys)
    plt.scatter(predict_x, predict_y, color = 'g')
    plt.plot(xs, regression_line)
    plt.show()
