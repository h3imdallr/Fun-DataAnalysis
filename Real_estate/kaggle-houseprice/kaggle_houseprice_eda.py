"""
- Obj: EDA for house pricing, data from kaggle
- ref:

https://www.kaggle.com/fiorenza2/journey-to-the-top-10
https://www.kaggle.com/tamatoa/house-prices-predicting-sales-price
https://www.kaggle.com/apapiu/regularized-linear-models
https://www.kaggle.com/dansbecker/learning-to-use-xgboost
(https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)

- Missions

* How to deal with data that categorical&numerical mixed
* Handling with Missing Value
* Normality , Skewness
* Outliers
* FEATURE SELECTION: http://scikit-learn.org/stable/modules/feature_selection.html
* Modeling:
    . regularized regression
    . xgBoost

"""


import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

# %matplotlib inline

# import data
df = pd.read_csv('./kaggle-houseprice/train.csv')

#check the number of datapoints
print(len(df))

"""
::: EDA :::
"""

#histogram
sns.distplot(df['OverallQual'],kde=False)
sns.distplot(df['SalePrice'],kde=False)

#skewness and kurtosis
print("Skewness: %f" % df['SalePrice'].skew())
print("Kurtosis: %f" % df['SalePrice'].kurt())

# check correlation btw OverQual and SalePrice
plt.figure()
sns.boxplot(df['OverallQual'], df['SalePrice'], color = 'steelblue')
plt.legend(); plt.xlabel('OverallQual'); plt.ylabel('SalePrice')
plt.savefig('./figures/hist_Qual_Price.png'); plt.close()

# for CATEGORICAL FEATURES
sns.boxplot(df['SaleCondition'], df['SalePrice'])
sns.boxplot(df['LotShape'], df['SalePrice'])

# Check Missing Values
print((df.isnull().sum()/len(df)).sort_values(ascending=False))

# Take out columns with too many NaNs
col_nan=(df.isnull().sum()/len(df)).sort_values(ascending=False)[:7].index
df = df.drop(col_nan, axis=1)

# replace with mean of each column for the rest of the features
df = df.fillna(df.mean())


"""
::: Feature Selection :::
"""

""" (1) feature select by correlation matrix """
# - the function automatically only applies for numerical data
corr_matrix = df.corr()
print (corr_matrix)

# take out most correlated features with 'SalePrice'
print (corr_matrix['SalePrice'])
print (corr_matrix['SalePrice'].nlargest(15))

drop_col = list(corr_matrix['SalePrice'].nlargest(15).index)
drop_col.remove('SalePrice')
print (drop_col)

# select only columns that concerned
selected_columns= [c for c in list(corr_matrix.columns) if c not in drop_col]
new_corr = corr_matrix[corr_matrix.columns.isin(selected_columns)]

# check with heatmap w/ new corr matrix
plt.figure()
sns.heatmap(new_corr, vmax=0.8, square=False)
plt.xticks(rotation=45, fontsize= 7);plt.yticks(rotation=45, fontsize= 7);
# plt.legend(); plt.xlabel(''); plt.ylabel('')
plt.savefig('./figures/heatmap.png'); plt.close()

# scatter plot
# sns.pairplot(df)

""" (2) feature select by K-Best """
# split categorical data and numerical data
num_df = df.select_dtypes(include=[np.number])
cat_df = df.select_dtypes(exclude=[np.number])
print(" - among total number of features (%d), "
      "\n -numerical features: %d \n -categorical features: %d"
      % (len(df.columns), len(num_df.columns), len(cat_df.columns)) )

# K-Best
# from sklearn.feature_selection import SelectKBest
# predictors = num_df.columns[:-1] # collect all features w/o target feature
# selector = SelectKBest(k=1)
# selector.fit(df[predictors], df['SalePrice'])
#
# scores = -np.log10(selector.pvalues_)
# plt.bar(range(len(scores)), scores)
# plt.xticks(np.arange(.2, len(scores)+.2), predictors, rotation="vertical")
#
#

""" outelieres?"""


""" Regularized Regression"""
""" XGBoost"""