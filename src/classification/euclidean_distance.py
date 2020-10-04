import warnings
from collections import Counter

import numpy as np
from matplotlib import style


def k_nearest_distance(data, predict, k = 3):
    if len(data) >= k:
        warnings.warn("K is set to a value less than total voting groups")

    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = sqrt(((features[0] - features[0]) ** 2) + (features[1] - features[1]) ** 2)
            """
            this is not so fast .. on large data set..
            also works with only two features ... hence we will change it to use `numpy` lib for the same
            """

            # euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict)) ** 2))
            """
            we will use more simple version of the same - available in the numpy library            
            """
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]  # this is the issue w/ this algo - it order of O(n)

    # print('Votes: ', votes)
    # print('Counter: ', Counter(votes))
    # print('Most Common: ', Counter(votes).most_common(1))

    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


if __name__ == '__main__':
    style.use('fivethirtyeight')

    # plot_1 = [1, 3]
    # plot_2 = [2, 5]
    # euclidean_distance = sqrt(((plot_1[0] - plot_2[0]) ** 2) + (plot_1[1] - plot_2[1]) ** 2)
    # print(euclidean_distance)

    dataset = {'g': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
    new_feature = [5, 7]

    # [[plt.scatter(ii[0], ii[1], s = 100, color = i) for ii in dataset[i]] for i in dataset]
    # plt.scatter(new_feature[0], new_feature[1], s = 100)
    # plt.show()

    result = k_nearest_distance(dataset, new_feature, k = 3)
    print('Result: ', result)
