import numpy as np
import scipy
import matplotlib
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("train.csv")


print(df.columns)