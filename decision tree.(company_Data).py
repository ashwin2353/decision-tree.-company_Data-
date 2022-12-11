# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:14:07 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_csv("Company_Data.csv")
df.shape
df.dtypes
df.info()

df.isnull().sum()
df.head()


df["ShelveLoc"].value_counts()
df['Urban'].value_counts()
df['US'].value_counts()
df.groupby("ShelveLoc").size()
#===============================
# finding duplicate rows
df.duplicated()
df[df.duplicated()] # there is no duplicates rows
len(df.duplicated())

#=================================
#finding duplicate columns
df.columns.duplicated() # there is no duplicates columns

#================================
import matplotlib.pyplot as plt
import seaborn as sns

for col in df:
    print(col)
    print(plt.scatter(df[col],df[col]))
#===============================================
# Labelencoding

from sklearn.preprocessing import LabelEncoder
SS = LabelEncoder()
df['ShelveLoc'] = SS.fit_transform(df['ShelveLoc'])
df['Urban'] = SS.fit_transform(df['Urban'])
df['US'] = SS.fit_transform(df['US'])
df.dtypes
#===============================================
# data spliting the variable as X and Y
Y = df['Sales']
Y.shape
X = df[['CompPrice','Income','Population','Price','ShelveLoc','Age','Urban','US']]
X.shape
list(X)


# Sub plot


# plt.subplot(#Total number of rows, total number of columns, plot number)
plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.scatter(X['CompPrice'],X['Advertising'],color= "black")

plt.subplot(1,2,2)
plt.scatter(X['Income'],X['Population'],color= "black")
#========================================================
# scatter plot
feature = ['CompPrice',
 'Income',
 'Advertising',
 'Population',
 'Price',
 'ShelveLoc',
 'Age',
 'Education',
 'Urban',
 'US']

list(enumerate(feature))

plt.figure(figsize = (15, 30))
for i in enumerate(feature):
    plt.subplot(4, 3,i[0]+1)
    plt.scatter(X[i[1]],X['Income'],color= "black")
    plt.xticks(rotation = 45)
#==========================================================
# histogram

plt.figure(figsize = (15, 30))
for i in enumerate(feature):
    plt.subplot(4, 3,i[0]+1)
    plt.hist(X[i[1]],color="black")
    plt.xticks(rotation = 45)

plt.hist(X["CompPrice"],color="black")
#======================================================
# boxplot

import numpy as np
plt.boxplot(df['Sales'],vert=False)

Q1 = np.percentile(df['Sales'],25)
Q2 = np.percentile(df['Sales'],50)
Q3 = np.percentile(df['Sales'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['Sales'] < LW) | (df['Sales'] > UW)]
len(df[(df['Sales'] < LW) | (df['Sales'] > UW)])
# 2 out layers(316,376)

#=========================================================
import numpy as np
plt.boxplot(df['CompPrice'],vert=False)

Q1 = np.percentile(df['CompPrice'],25)
Q2 = np.percentile(df['CompPrice'],50)
Q3 = np.percentile(df['CompPrice'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['CompPrice'] < LW) | (df['CompPrice'] > UW)]
len(df[(df['CompPrice'] < LW) | (df['CompPrice'] > UW)])

# 2 out layers (42,310)
#======================================================
plt.boxplot(df['Income'],vert=False)

Q1 = np.percentile(df['Income'],25)
Q2 = np.percentile(df['Income'],50)
Q3 = np.percentile(df['Income'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['Income'] < LW) | (df['Income'] > UW)]

len(df[(df['Income'] < LW) | (df['Income'] > UW)])
# 0 outlayers
#=====================================================
plt.boxplot(df['Advertising'],vert=False)

Q1 = np.percentile(df['Advertising'],25)
Q2 = np.percentile(df['Advertising'],50)
Q3 = np.percentile(df['Advertising'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['Advertising'] < LW) | (df['Advertising'] > UW)]

len(df[(df['Advertising'] < LW) | (df['Advertising'] > UW)])
# 0 outlayers

#=====================================================
plt.boxplot(df['Population'],vert=False)

Q1 = np.percentile(df['Population'],25)
Q2 = np.percentile(df['Population'],50)
Q3 = np.percentile(df['Population'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['Population'] < LW) | (df['Population'] > UW)]

len(df[(df['Population'] < LW) | (df['Population'] > UW)])
# 0 outlayers

#==================================================
plt.boxplot(df['Price'],vert=False)

Q1 = np.percentile(df['Price'],25)
Q2 = np.percentile(df['Price'],50)
Q3 = np.percentile(df['Price'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['Price'] < LW) | (df['Price'] > UW)]

len(df[(df['Price'] < LW) | (df['Price'] > UW)])
# 5 out layers (42,125,165,174,367)
#=================================================
plt.boxplot(df['Age'],vert=False)

Q1 = np.percentile(df['Age'],25)
Q2 = np.percentile(df['Age'],50)
Q3 = np.percentile(df['Age'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['Age'] < LW) | (df['Age'] > UW)]

len(df[(df['Age'] < LW) | (df['Age'] > UW)])
# 0 outlayers

#===============================================
plt.boxplot(df['Education'],vert=False)

Q1 = np.percentile(df['Education'],25)
Q2 = np.percentile(df['Education'],50)
Q3 = np.percentile(df['Education'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['Education'] < LW) | (df['Education'] > UW)]

len(df[(df['Education'] < LW) | (df['Education'] > UW)])
# 0 outlayers
#=================================================
# removing out layers from the data
df = df.drop([316,376,42,310,125,165,174,367])
df.shape

#===================================================
# data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

#===================================================
# model fitting
from sklearn.tree import DecisionTreeRegressor
Regressor = DecisionTreeRegressor(max_depth=3)
Regressor.fit(X_train,Y_train)

Y_pred_train = Regressor.predict(X_train)
Y_pred_test = Regressor.predict(X_test)

Regressor.tree_.max_depth
Regressor.tree_.node_count
#====================================================
# Tree visualization
# pip install graphviz

from sklearn import tree
import graphviz

dot_data = tree.export_graphviz(Regressor,out_file=None, filled=True, rounded=True, special_characters=True)

graph = graphviz.source(dot_data)
graph

#=====================================================
# metrics
from sklearn.metrics import mean_squared_error

print("Training Error",mean_squared_error(Y_train,Y_pred_train).round(3))
print("testing Error",mean_squared_error(Y_test,Y_pred_test).round(3))
#=========================================================

# regularization(Ridge)
from sklearn.linear_model import Ridge
Rg = Ridge(alpha = 30)
Rg.fit(X_train,Y_train)
y_pred_train = Rg.predict(X_train)
y_pred_test = Rg.predict(X_test)


from sklearn.metrics import mean_squared_error
Training_error = mean_squared_error(Y_train,Y_pred_train)
Test_error = mean_squared_error(Y_test,Y_pred_test)

print("Training error",Training_error.round(3))
print("Test error",Test_error.round(3))


Rg.coef_

pd.DataFrame(Rg.coef_)
pd.DataFrame(X.columns)
d1 = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(Rg.coef_)],axis = 1)
d1

#==============================================
# Regularization (Lasso)
from sklearn.linear_model import Lasso
LaR = Lasso(alpha = 8)
LaR.fit(X_train,Y_train)
y_pred_train = LaR.predict(X_train)
y_pred_test = LaR.predict(X_test)


from sklearn.metrics import mean_squared_error

print("Training error:",mean_squared_error(Y_train,Y_pred_train).round(3))
print("Test error:",mean_squared_error(Y_test,Y_pred_test).round(3))

LaR.coef_

pd.DataFrame(LaR.coef_)
pd.DataFrame(X.columns)
d2 = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(LaR.coef_)],axis = 1)
d2















