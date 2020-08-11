# import pacakegs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pylab as plt
import seaborn as sns
# print dasetset
dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/14. Random Forests - Done/Fraud_check.csv")
print(dataset.head())
colnames = list(dataset.columns)

# change str data type into int data type
dataset.loc[dataset['Undergrad']=='YES','Undergrad']=1
dataset.loc[dataset['Undergrad']=='NO','Undergrad']=0

dataset.loc[dataset['Marital.Status']=='Single','Marital.Status']=1
dataset.loc[dataset['Marital.Status']=='Married','Marital.Status']=2
dataset.loc[dataset['Marital.Status']=='Divorced','Marital.Status']=3

dataset.loc[dataset['Urban']=='YES','Urban']=1
dataset.loc[dataset['Urban']=='NO','Urban']=0

dataset.loc[dataset['Taxable.Income']<=30000,'Taxable.Income']=1
dataset.loc[dataset['Taxable.Income']>30000,'Taxable.Income']=0

#plot bar graph
fc = dataset['Taxable.Income'].value_counts()
plt.hist(dataset['Taxable.Income'],edgecolor='k')
for i, v in enumerate(fc):
    plt.text(i,
              v,
             fc[i],
              fontsize=18,
              color="red")
plt.xlabel('Taxable Income')
plt.show()

ug = dataset['Undergrad'].value_counts()
plt.hist(dataset['Undergrad'],edgecolor='k')
for i, v in enumerate(ug):
    plt.text(i,
              v,
             ug[i],
              fontsize=18,
              color="red")
plt.xlabel('Undergrad')
plt.show()

ur = dataset['Urban'].value_counts()
plt.hist(dataset['Urban'],edgecolor='k')
for i, v in enumerate(ur):
    plt.text(i,
              v,
             ur[i],
              fontsize=18,
              color="red")
plt.xlabel('Urban')
plt.show()

# check skewnee & kurtosis
# graph plot
print("City Population Skewness : \n",dataset['City.Population'].skew(),'\n\n\n')
print("City Population Kurtosis : \n",dataset['City.Population'].kurt(),'\n\n\n')

plt.hist(dataset['City.Population'],edgecolor='k')
plt.xlabel('City Population')
plt.show()
sns.distplot(dataset['City.Population'],hist=False)
plt.xlabel('City Population')
plt.show()
plt.boxplot(dataset['City.Population'])
plt.xlabel('City Population')
plt.show()


print("Work Experience Skewness : \n",dataset['Work.Experience'].skew(),'\n\n\n')
print("Work Experience Kurtosis : \n",dataset['Work.Experience'].kurt(),'\n\n\n')

plt.hist(dataset['Work.Experience'],edgecolor='k')
plt.xlabel('Work Experience')
plt.show()
sns.distplot(dataset['Work.Experience'],hist=False)
plt.xlabel('Work Experience')
plt.show()
plt.boxplot(dataset['Work.Experience'])
plt.xlabel('Work Experience')
plt.show()

sns.pairplot(dataset)
plt.show()

# pie graph plot
dataset.Undergrad.value_counts().plot(kind="pie")
plt.show()

dataset.Urban.value_counts().plot(kind="pie")
plt.show()

#split train & test dataset
y = dataset['Taxable.Income']
x = dataset.drop(['Taxable.Income'],axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#convert datatype into int.
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
