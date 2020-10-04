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


class SupportVectorMachine:

    def __init__(self, visualization = True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

        self.data = None
        self.max_feature_value = None
        self.min_feature_value = None

    # training of data
    def fit(self, data):
        self.data = data

        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        # temporary creating to find min and max value
        all_data = []
        for yi in self.data:
            for feature_set in self.data[yi]:
                for feature in feature_set:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [
            self.max_feature_value * 0.1,
            self.max_feature_value * 0.01,
            self.max_feature_value * 0.001,  # point of expense:
        ]

        b_range_multiple = 5  # extremely expensive
        b_multiple = 5

        latest_optimum = self.max_feature_value * 10
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])

            optimized = False  # we can do this because convex; we know the optimum point
            while not optimized:
                pass

    def predict(self, features):
        # sign( x.w + b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        return classification
