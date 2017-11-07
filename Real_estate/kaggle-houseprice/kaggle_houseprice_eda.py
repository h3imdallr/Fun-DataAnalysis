"""
- Obj: EDA for house pricing, data from kaggle
- ref:
https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
https://www.kaggle.com/apapiu/regularized-linear-models
https://www.kaggle.com/tamatoa/house-prices-predicting-sales-price
https://www.kaggle.com/fiorenza2/journey-to-the-top-10
https://www.kaggle.com/dansbecker/learning-to-use-xgboost
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

# def hist_plot(in_x, savepath):
#     plt.figure(figsize=figsize)
#     sns.distplot(in_x,  alpha=alpha, label=label)
#     plt.legend(); plt.xlabel(''); plt.ylabel('')
#     plt.savefig(savepath=savepath, dpi=300);plt.close()

#histogram
sns.distplot(df['OverallQual'],kde=False)
sns.distplot(df['SalePrice'],kde=False)

#skewness and kurtosis
print("Skewness: %f" % df['SalePrice'].skew())
print("Kurtosis: %f" % df['SalePrice'].kurt())

# check correlation btw OverQual and SalePrice
# plt.scatter(df['OverallQual'], df['SalePrice'], color = 'steelblue')
sns.boxplot(df['OverallQual'], df['SalePrice'], color = 'steelblue')

# :: Feature Engineering ::

# check correlation
# 1) correlation matrix
corr_matrix = df.corr()
print (corr_matrix)
print (corr_matrix['SalePrice'])
print (corr_matrix['SalePrice'].nlargest(7))

#take out most correlated


# # 1) heatmap
# sns.heatmap(corr_matrix, vmax=0.8, square=False)
# plt.xticks(rotation=45, fontsize= 7);plt.yticks(rotation=45, fontsize= 7);
#
# # 2) scatter
# sns.pairplot(df)
#
#


