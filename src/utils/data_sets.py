import pandas as pd

from settings import DATA_SET_PATH


def load_breast_cancer_wisconsin_data():
    """
    Source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
    """
    df = pd.read_csv(f'{DATA_SET_PATH}/breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace = True)
    df.drop(['id'], 1, inplace = True)
    return df
