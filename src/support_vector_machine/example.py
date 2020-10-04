import logging

import numpy as np
from pandas import DataFrame
from sklearn import svm
from sklearn.model_selection import train_test_split

from settings import TRANSIENT_DIR_PATH
from utils.commons import init_logging
from utils.data_sets import load_breast_cancer_wisconsin_data
from utils.file import get_pickled_object, pickleize_object

CACHE_CLASSIFIER = False


def get_svm_classifier(name: str, df: DataFrame):
    file_path = f'{TRANSIENT_DIR_PATH}/{name}.svm.classifier.pickle'

    if CACHE_CLASSIFIER:
        classifier = get_pickled_object(file_path)
        if classifier is not None:
            logger.debug("fetching pickled classifier")
            return classifier

    logger.debug("creating new classifier")

    # defining features and classes
    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    # training and test data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    classifier = svm.SVC()
    classifier.fit(X_train, y_train)  # training the classifier

    accuracy = classifier.score(X_test, y_test)  # testing the classifier
    print("'SVM' accuracy: ", accuracy)

    if CACHE_CLASSIFIER:
        pickleize_object(file_path, classifier)

    return classifier


if __name__ == '__main__':
    init_logging()
    logger = logging.getLogger('MLTrials')

    _df = load_breast_cancer_wisconsin_data()
    knn_classifier = get_svm_classifier('breast-cancer-wisconsin', _df)

    # examples_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 3, 1, 1, 1, 3, 2, 1]])
    # examples_measures = examples_measures.reshape(len(examples_measures), -1)  # reshaping data
    #
    # prediction = knn_classifier.predict(examples_measures)
    #
    # print('Predictions: ', prediction)
