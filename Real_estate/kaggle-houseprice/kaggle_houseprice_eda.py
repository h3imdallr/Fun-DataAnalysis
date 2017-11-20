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
* Handling with Missing Value: http://www.bogotobogo.com/python/scikit-learn/scikit_machine_learning_Data_Preprocessing-Missing-Data-Categorical-Data.php
* Normality , Skewness
* Outliers
* FEATURE SELECTION: http://scikit-learn.org/stable/modules/feature_selection.html
* Modeling:
    . regularized regression : https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/

"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import norm

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
sns.distplot(df['OverallCond'],kde=False)

# Normality
# - check with skewness and kurtosi
print("\n- Skewness of SalePrice histogram: %f" % df['SalePrice'].skew())
print("\n- Kurtosis of SalePrice histogram: %f" % df['SalePrice'].kurt())
# - check with Q-Q Plot
stats.probplot(df['SalePrice'], plot=plt)
# - normalize
sns.distplot(np.log(df['SalePrice']),kde=True)
stats.probplot(np.log(df['SalePrice']), plot=plt)
# sm.qqplot(df['SalePrice'], line='q')

# check correlation btw OverQual and SalePrice
plt.figure()
sns.boxplot(df['OverallQual'], df['SalePrice'], color = 'steelblue')
plt.legend(); plt.xlabel('OverallQual'); plt.ylabel('SalePrice')
plt.savefig('./figures/hist_Qual_Price.png'); plt.close()

# for CATEGORICAL FEATURES
sns.boxplot(df['SaleCondition'], df['SalePrice'])
sns.boxplot(df['LotShape'], df['SalePrice'])
sns.boxplot(df['OverallQual'], df['SalePrice'])

# Check Missing Values
print("\n - Missinv Value portion: \n", (df.isnull().sum()/len(df)).sort_values(ascending=False))

# Take out columns with too many NaNs
col_nan=(df.isnull().sum()/len(df)).sort_values(ascending=False)[:7].index
df = df.drop(col_nan, axis=1)

# replace with mean of each column for the rest of the features
df = df.fillna(df.mean())

""" outelieres?"""


"""
::: Feature Selection :::
"""

""" (1) feature select by correlation matrix """
# - the function automatically only applies for numerical data
corr_matrix = df.corr()
print (corr_matrix)

# take out most correlated features with 'SalePrice'
print ("-Correlation Matrix:\n", corr_matrix['SalePrice'])
print ("-Most relavant features by correlation matrix:\n",
       corr_matrix['SalePrice'].nlargest(11).index)

#new correlation matrix with selected features
new_corr = df[corr_matrix['SalePrice'].nlargest(11).index].corr()

# check with heatmap w/ new corr matrix
# highly correlated variables can be taken out among 'SalePrice' features
plt.figure()
sns.heatmap(new_corr, vmax=0.8, square=True)
plt.xticks(rotation=45, fontsize= 7);plt.yticks(rotation=45, fontsize= 7);
# plt.legend(); plt.xlabel(''); plt.ylabel('')
# plt.savefig('./figures/heatmap.png'); plt.close()


if False:
      """ when you want to opt out features with less dependant  """
      drop_col = list(corr_matrix['SalePrice'].nlargest(28).index)
      drop_col.remove('SalePrice')
      print (drop_col)

      # select only columns that concerned
      selected_columns= [c for c in list(corr_matrix.columns) if c not in drop_col]
      new_corr = df[selected_columns].corr()
      print ("- most relavant features by correlation matrix: ", new_corr.columns)

      # check with heatmap w/ new corr matrix
      plt.figure()
      sns.heatmap(new_corr, vmax=0.8, square=True)
      plt.xticks(rotation=45, fontsize= 7);plt.yticks(rotation=45, fontsize= 7);
      # plt.legend(); plt.xlabel(''); plt.ylabel('')
      plt.savefig('./figures/heatmap.png'); plt.close()

      # scatter plot
      # sns.pairplot(df)


""" (2) feature select by K-Best """
# split categorical data and numerical data
num_df = df.select_dtypes(include=[np.number])
cat_df = df.select_dtypes(exclude=[np.number])
print("\n- among total number of features (%d), "
      "\n-numerical features: %d \n-categorical features: %d"
      % (len(df.columns), len(num_df.columns), len(cat_df.columns)) )

# K-Best
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
predictors = num_df.columns[:-1] # collect all features w/o target feature
selection = SelectKBest(f_regression, k=5).fit(df[predictors], df['SalePrice'])
# selection = SelectKBest(mutual_info_regression,k=5).fit(df[predictors], df['SalePrice'])

scores = -np.log(selection.pvalues_)# scores = selection.scores_

plt.figure()
plt.bar(range(len(scores)), scores)
plt.xticks(np.arange(.2, len(scores)+.2), predictors, rotation="vertical"); plt.tight_layout()
plt.savefig('./figures/K-Best_score.png'); plt.close()

# use top 10 most relavant features
scores_sr = pd.Series(scores,index=predictors)
selected_features = scores_sr.nlargest(10)
print ("- Most relavant features by selectKbest(sklearn): \n", selected_features.index)


""" (3) ANOVA for categorical data """
"""
- The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean.
- The ANOVA test has important assumptions that must be satisfied in order for the associated p-value to be valid.
a. The samples are independent.
b. Each sample is from a normally distributed population.
c. The population standard deviations of the groups are all equal. This property is known as homoscedasticity.
"""
#we use list compression to assign our return values to cat and for simplicity
#Read more about this technique and you will enjoy its power
col_cat=list(cat_df.columns)
#print(cat)
def anova_test(inDF):
    anv = pd.DataFrame()
    anv['features'] = col_cat
    pvals=[]
    for c in col_cat:
        samples=[]
        for cls in inDF[c].unique():
            s=inDF[inDF[c]==cls]['SalePrice'].values
            samples.append(s)
        pval=stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv["pval"]=pvals
    return anv.sort_values("pval")

cat_df['SalePrice']=df.SalePrice.values
k=anova_test(cat_df)
k['disparity']=np.log(1./k['pval'].values)

plt.figure()
sns.barplot(data=k,x="features",y="disparity")
plt.xticks(rotation=90); plt.tight_layout()
plt.savefig("./figures/ANOVA_categoricalfeature.png");plt.close()


"""
::: Model fitting :::
"""
"""
(0) Preprocessing
- normalize the skewed data
(** note that non-linear regression model may not need normality guaranteed  )
- deal with missing values(done)
- deal with outliers
"""
# unskew all data
print(num_df.skew())
skewed_features = num_df.columns[num_df.skew()>0.75]

# *** note that np.log1p() is used, not np.log() ***
# if np.log() is used, the dataframe is filled with -inf
# http://rfriend.tistory.com/295
num_df_norm = np.log1p(num_df[skewed_features]) # ---***

X_train  = num_df_norm.drop('SalePrice', axis=1)
y = num_df_norm['SalePrice']

""" (1)-a Regularized Regression : LASSO """
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import cross_val_score

model_lasso = LassoCV(cv=20).fit(X_train, y)
print("LASSO/alpha: ",model_lasso.alpha_)
print("LASSO/Coef: ",model_lasso.coef_)
print("LASSO/R^2 Score: ",model_lasso.score(X_train,y))


""" (1)-b Regularized Regression : Elastic Net """
model_elastic = ElasticNetCV(cv=20,random_state=0).fit(X_train,y)
print("ElasticNet/alpha: ",model_elastic.alpha_)
print("ElasticNet/Coef: ",model_elastic.coef_)
print("ElasticNet/R^2 Score: ",model_elastic.score(X_train,y))


""" (2) XGBoost """



""" categorical predictors ???
- 1) one-hot encoding으로 coefficient를 구할 수 있음: https://datascienceschool.net/view-notebook/7dda1bc9ad1c435fb309ea88f672eff9/
- OR 2) ANOVA:https://datascienceschool.net/view-notebook/a60e97ad90164e07ad236095ca74e657/
"""