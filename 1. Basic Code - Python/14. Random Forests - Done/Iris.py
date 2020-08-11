# import pacakegs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pylab as plt
import seaborn as sns
# print dasetset
dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/14. Random Forests - Done/iris.csv")
print(dataset.head())
colnames = list(dataset.columns)
print(colnames,'\n\n')

# pie graph plot
dataset.Species.value_counts().plot(kind="pie")
plt.show()

dataset.loc[dataset['Species']=='setosa','Species']=1
dataset.loc[dataset['Species']=='versicolor','Species']=2
dataset.loc[dataset['Species']=='virginica','Species']=3


# check skewnee & kurtosis
# graph plot
print("Sepal Length Skewness : \n",dataset['Sepal.Length'].skew(),'\n\n\n')
print("Sepal Length Kurtosis : \n",dataset['Sepal.Length'].kurt(),'\n\n\n')

plt.hist(dataset['Sepal.Length'],edgecolor='k')
plt.xlabel('Sepal Length')
plt.show()
sns.distplot(dataset['Sepal.Length'],hist=False)
plt.xlabel('Sepal Length')
plt.show()
plt.boxplot(dataset['Sepal.Length'])
plt.xlabel('Sepal Length')
plt.show()


print("Sepal Width Skewness : \n",dataset['Sepal.Width'].skew(),'\n\n\n')
print("Sepal Width Kurtosis : \n",dataset['Sepal.Width'].kurt(),'\n\n\n')

plt.hist(dataset['Sepal.Width'],edgecolor='k')
plt.xlabel('Sepal Width')
plt.show()
sns.distplot(dataset['Sepal.Width'],hist=False)
plt.xlabel('Sepal Width')
plt.show()
plt.boxplot(dataset['Sepal.Width'])
plt.xlabel('Sepal Width')
plt.show()


print("Petal Length Skewness : \n",dataset['Petal.Length'].skew(),'\n\n\n')
print("Petal Length Kurtosis : \n",dataset['Petal.Length'].kurt(),'\n\n\n')

plt.hist(dataset['Petal.Length'],edgecolor='k')
plt.xlabel('Petal Length')
plt.show()
sns.distplot(dataset['Petal.Length'],hist=False)
plt.xlabel('Petal Length')
plt.show()
plt.boxplot(dataset['Petal.Length'])
plt.xlabel('Petal Length')
plt.show()


print("Petal Width Skewness : \n",dataset['Petal.Width'].skew(),'\n\n\n')
print("Petal Width Kurtosis : \n",dataset['Petal.Width'].kurt(),'\n\n\n')

plt.hist(dataset['Petal.Width'],edgecolor='k')
plt.xlabel('Petal Width')
plt.show()
sns.distplot(dataset['Petal.Width'],hist=False)
plt.xlabel('Petal Width')
plt.show()
plt.boxplot(dataset['Petal.Width'])
plt.xlabel('Petal Width')
plt.show()

sns.pairplot(dataset)
plt.show()


#split train & test dataset
y = dataset['Species']
x = dataset.drop(['Species'],axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=42)

#convert datatype into int.
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
