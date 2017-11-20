"""
- https://www.kaggle.com/dansbecker/learning-to-use-xgboost
- https://brunch.co.kr/@snobberys/137

. evaluation
- https://machinelearningmastery.com/evaluate-gradient-boosting-models-xgboost-python/
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

df = pd.read_csv('./kaggle-houseprice/train.csv')
df = df.dropna(axis=0, subset=['SalePrice'])

# Take out columns with too many NaNs
col_nan=(df.isnull().sum()/len(df)).sort_values(ascending=False)[:7].index
df = df.drop(col_nan, axis=1)

# replace with mean of each column for the rest of the features
df = df.fillna(df.mean())

X = df.select_dtypes(include=[np.number]).drop(['SalePrice'],axis=1).fillna(df.mean())
y = df['SalePrice']
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

# my_imputer = Imputer()
# train_X = my_imputer.fit_transform(train_X)
# test_X = my_imputer.transform(test_X)

from xgboost import XGBRegressor
model_XGboost = XGBRegressor().fit(train_X, train_y, verbose=False)


# from sklearn.metrics import mean_absolute_error
# predictions = model_XGboost.predict(test_X)
# print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
"""accuracy"""
