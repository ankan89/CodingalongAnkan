# import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import matplotlib.pylab as plt
import seaborn as sns
# print dataset
dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/14. Random Forests - Done/Company_Data.csv")

print(dataset.head())

colnames = list(dataset.columns)

# change str data type into unique value int data type
dataset.loc[dataset['ShelveLoc']=='Good','ShelveLoc']=0
dataset.loc[dataset['ShelveLoc']=='Bad','ShelveLoc']=1
dataset.loc[dataset['ShelveLoc']=='Medium','ShelveLoc']=2

dataset.loc[dataset['Urban']=='Yes','Urban']=1
dataset.loc[dataset['Urban']=='No','Urban']=0

dataset.loc[dataset['US']=='Yes','US']=1
dataset.loc[dataset['US']=='No','US']=0

# value count
dataset['Sales'].value_counts()

#plot bar graph
co = dataset['ShelveLoc'].value_counts()
plt.hist(dataset['ShelveLoc'],edgecolor='k')
plt.xlabel("ShelveLoc")
plt.grid()
plt.show()

u = dataset['US'].value_counts()
plt.hist(dataset['US'],edgecolor='k')
plt.xlabel("US")
plt.grid()
plt.show()

ur = dataset['Urban'].value_counts()
plt.hist(dataset['Urban'],edgecolor='k')
plt.xlabel("Urban")
plt.grid()
plt.show()

# check skewnee, kurtosis & plot graph
# sales
print(dataset['Sales'].skew(),'\n\n')
print(dataset['Sales'].kurt(),'\n\n')

plt.hist(dataset['Sales'],edgecolor='k')
plt.xlabel("Sales")
plt.show()
sns.distplot(dataset['Sales'],hist=False)
plt.xlabel("Sales")
plt.show()
plt.boxplot(dataset['Sales'])
plt.xlabel("Sales")
plt.show()

# compprice
print(dataset['CompPrice'].skew(),'\n\n')
print(dataset['CompPrice'].kurt(),'\n\n')
plt.hist(dataset['CompPrice'],edgecolor='k')
plt.xlabel("CompPrice")
plt.show()
sns.distplot(dataset['CompPrice'],hist=False)
plt.xlabel("CompPrice")
plt.show()
plt.boxplot(dataset['CompPrice'])
plt.xlabel("CompPrice")
plt.show()

# population
print(dataset['Population'].skew(),'\n\n')
print(dataset['Population'].kurt(),'\n\n')
plt.hist(dataset['Population'],edgecolor='k')
plt.xlabel("Population")
plt.show()
sns.distplot(dataset['Population'],hist=False)
plt.xlabel("Population")
plt.show()
plt.boxplot(dataset['Population'])
plt.xlabel("Population")
plt.show()

# pair plot
sns.pairplot(dataset)
plt.show()

# pie graph plot
dataset.ShelveLoc.value_counts().plot(kind="pie")
plt.show()
dataset.Urban.value_counts().plot(kind="pie")
plt.show()

#split train & test dataset
y = dataset['Sales']
x = dataset.drop(['Sales'],axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# convert datatype into int.
X_train = X_train.astype("int")
Y_train = Y_train.astype("int")
Y_test = Y_test.astype("int")

#implement random forest classifier with different hyperparameter.
# n_estimators=50,criterion='entropy',max_features='sqrt'
clf=RandomForestClassifier(n_estimators=50,criterion='entropy',max_features='sqrt')
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
score=metrics.balanced_accuracy_score(Y_test, Y_pred)
print("Accuracy Score 1 : ",score,'\n')

# n_estimators=100,criterion='entropy',max_features='log2'
clf1=RandomForestClassifier(n_estimators=100,criterion='entropy',max_features='log2')
clf1.fit(X_train,Y_train)
Y_pred=clf1.predict(X_test)
score1=metrics.balanced_accuracy_score(Y_test, Y_pred)
print("Accuracy Score 2 : ",score1,'\n')

# n_estimators=100,max_features='log2',max_depth=100
clf2=RandomForestClassifier(n_estimators=100,max_features='log2',max_depth=100)
clf2.fit(X_train,Y_train)
Y_pred=clf2.predict(X_test)
score2=metrics.balanced_accuracy_score(Y_test, Y_pred)
print("Accuracy Score 3 : ",score2,'\n')

# Adaboost classifier
# n_estimators=100, random_state=0
clf3 = AdaBoostClassifier(n_estimators=100, random_state=0)
clf3.fit(X_train,Y_train)
Y_pred=clf3.predict(X_test)
score2=metrics.balanced_accuracy_score(Y_test, Y_pred)
print("Accuracy Score 4 : ",score2,'\n')
