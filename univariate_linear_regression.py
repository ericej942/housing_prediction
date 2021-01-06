import numpy as np
import scipy
import matplotlib
import pandas as pd
from sklearn.linear_model import LinearRegression
import csv

## Make Training Data
train = pd.read_csv("train.csv")
train = pd.get_dummies(train)

## Make Testing Data
test = pd.read_csv("test.csv")
test = pd.get_dummies(test)


x = np.array(train["YearBuilt"]).reshape((-1, 1))  # predictors
y = np.array(train['SalePrice'])  # response

## Create Baseline Metric
null_rmse = np.sqrt(np.mean((np.mean(y) - y)**2))
print("Null Model RMSE: \t", null_rmse)

## Fit the Linear Rergression Model
mdl = LinearRegression().fit(x, y)

ids = test['Id']

## Measure the Training RMSE
train_pred = mdl.predict(x)
rmse = np.sqrt(np.mean((train_pred - y)**2))
print("Univariate Model RMSE: \t", rmse)

## Use test data to Predict Response
test_data = np.array(test["YearBuilt"]).reshape((-1, 1))
pred = mdl.predict(test_data)

## Write the results to a CSV
with open('submission01.csv', mode='w') as sub_file:
    writer = csv.writer(sub_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Id", "SalePrice"])

    for id in range(len(pred)):
        writer.writerow([ids[id], pred[id]])
