import logging
import random

from classification.euclidean_distance import k_nearest_distance
from utils.commons import init_logging
from utils.data_sets import load_breast_cancer_wisconsin_data

init_logging()
logger = logging.getLogger('MLTrials')

_df = load_breast_cancer_wisconsin_data()
full_data = _df.astype(float).values.tolist()  # making sure all data is float and nothing is string or char...
random.shuffle(full_data)  # in-place shuffle

test_size = 0.2

train_set = {2: [], 4: []}
test_set = {2: [], 4: []}

split_idx = int(test_size * len(full_data))

# print('Total Data: ', len(full_data))
# print('Split idx: ', split_idx)

train_data = full_data[:-split_idx]  # first {test_size}%
test_data = full_data[-split_idx:]  # last {test_size}%

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, _ = k_nearest_distance(train_set, data, k = 5)
        if group == vote:
            correct += 1
        total += 1

print("Accuracy: ", correct / total)
