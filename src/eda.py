import matplotlib.pyplot as plt
import pandas
from sklearn import datasets

df = pd.read_table("house-votes-84.data", sep=",", header=None, na_values="?")