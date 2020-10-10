import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

from utils.commons import init_logging

style.use('ggplot')

init_logging()
logger = logging.getLogger('MLTrials')

data_dict = {
    -1: np.array([[1, 7], [2, 8], [3, 8]]),
    1: np.array([[5, 1], [6, -1], [7, 3]])
}


def _find_boundaries(data):
    all_data = []  # temporary creating to find min and max value
    for yi in data:
        for feature_set in data[yi]:
            for feature in feature_set:
                all_data.append(feature)

    max_feature_value = max(all_data)
    min_feature_value = min(all_data)
    return min_feature_value, max_feature_value


class SupportVectorMachine:

    def __init__(self, visualization = True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

        self.data = None
        self.w = None
        self.b = None
        self.max_feature_value = None
        self.min_feature_value = None

    # training of data
    def fit(self, data):
        self.data = data

        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        self.min_feature_value, self.max_feature_value = _find_boundaries(self.data)

        # support vectors till we hit yi(xi.w+b) = ~1 we keep on stepping

        step_sizes = [
            self.max_feature_value * 0.1,
            self.max_feature_value * 0.01,
            # point of expense:
            self.max_feature_value * 0.001,
            # self.max_feature_value * 0.0001,
        ]

        b_range_multiple = 3  # alt:  5   # extremely expensive
        b_multiple = 5  # we don't need to take as small of steps with `b` as we do for `w`
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False  # we can do this because convex; we know the optimum point
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                        self.max_feature_value * b_range_multiple, step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # this is the issue with SVM; SMO is the fix
                        # yi ( xi.w + b ) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                if not i * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    # break

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print("Optimized a step")
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

        for i in self.data:
            for xi in self.data[i]:
                print(xi, ':', i * (np.dot(self.w, xi) + self.b))

    def predict(self, features):
        # sign( x.w + b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s = 200, marker = '*', c = self.colors[classification])
        else:
            print('feature set', features, 'is on the decision boundary')
        return classification

    def visualize(self):
        # scattering known feature sets
        [[self.ax.scatter(x[0], x[1], s = 100, color = self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane: v = x.w+b
        # where psv = 1 , nsv = -1, decision boundary = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        data_range = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min, hyp_x_max = data_range

        psv_min = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv_max = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv_min, psv_max], "k")

        nsv_min = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv_max = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv_min, nsv_max], "k")

        db_min = hyperplane(hyp_x_min, self.w, self.b, 0)
        db_max = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db_min, db_max], "g--")

        plt.show()
        return


if __name__ == '__main__':
    inp_data = {
        -1: np.array([[1, 7], [2, 8], [3, 8], ]),
        1: np.array([[5, 1], [6, -1], [7, 3], ])
    }
    svm = SupportVectorMachine()
    svm.fit(data = inp_data)
    # svm.visualize()

    predict_us = [[0, 10], [1, 3], [3, 4], [3, 5], [5, 5], [5, 6], [6, -5], [5, 8]]
    for p in predict_us:
        svm.predict(p)

    svm.visualize()
