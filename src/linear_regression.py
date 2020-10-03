import matplotlib.pyplot as plt
import numpy as np

from utils.commons import best_fit_slope_and_intercept

PLOT_XY = False
PLOT_XY_PREDICT = True

xs = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)
ys = np.array([2, 6, 1, 8, 9, 4], dtype = np.float64)

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

if PLOT_XY_PREDICT:
    plt.scatter(xs, ys)
    plt.scatter(predict_x, predict_y, color = 'g')
    plt.plot(xs, regression_line)
    plt.show()
