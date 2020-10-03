import matplotlib.pyplot as plt
from matplotlib import style

from utils.commons import best_fit_slope_and_intercept, coefficient_of_determination, create_dataset

style.use('fivethirtyeight')

PLOT_XY = False
PLOT_XY_PREDICT = True

xs, ys = create_dataset(99, 25, 2, correlation = 'pos')

if PLOT_XY:
    plt.scatter(xs, ys)
    plt.show()

m, b = best_fit_slope_and_intercept(xs, ys)

print("Slope: ", m)
print("Intercept: ", b)

# model: y = mx + b
regression_line = [(m * x) + b for x in xs]

predict_x = 100
predict_y = m * predict_x + b

r_squared = coefficient_of_determination(ys, regression_line)

print("R-squared: ", r_squared)

if PLOT_XY_PREDICT:
    plt.scatter(xs, ys)
    plt.scatter(predict_x, predict_y, s = 100, color = 'g')
    plt.plot(xs, regression_line)
    plt.show()
