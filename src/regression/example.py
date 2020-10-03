import math

import numpy as np
import quandl
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def load_sample_data_for_linear_regression():
    df = quandl.get('WIKI/GOOGL')
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

    forecast_col = 'Adj. Close'
    forecast_out = int(math.ceil(0.1 * len(df)))  # no. of days in future

    print("days in advance: ", forecast_out)

    # giving default to the data without any defined values
    df.fillna(-99999, inplace = True)

    # will get `forecast_out` days shifted
    df['label'] = df[forecast_col].shift(-forecast_out)

    return df, forecast_col, forecast_out


if __name__ == '__main__':
    _df, fc, fo = load_sample_data_for_linear_regression()

    # dropping data with no future data
    _df.dropna(inplace = True)

    # features = X; label = y
    X = np.array(_df.drop('label', 1))
    y = np.array(_df['label'])

    # scaling features...
    X = preprocessing.scale(X)

    # training and test data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    lr_classifier = LinearRegression()
    lr_classifier.fit(X_train, y_train)  # training the classifier

    lr_accuracy = lr_classifier.score(X_test, y_test)  # testing the classifier
    print("'Linear Regression' accuracy: ", lr_accuracy)  # squared error

    # trying SVM

    svm_classifier = svm.SVR()
    svm_classifier.fit(X_train, y_train)
    svm_accuracy = svm_classifier.score(X_test, y_test)
    print("'SVM' accuracy: ", svm_accuracy)

    # trying SVM w/ kernel

    svm_classifier = svm.SVR(kernel = 'poly')
    svm_classifier.fit(X_train, y_train)
    svm_accuracy = svm_classifier.score(X_test, y_test)
    print("'SVM w/ kernel' accuracy: ", svm_accuracy)
