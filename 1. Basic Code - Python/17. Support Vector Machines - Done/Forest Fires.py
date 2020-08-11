
# import pacakages.
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#imporrt dataset.
#print head,columns,describe of th e dataset.
forest = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/17. Support Vector Machines - Done/forestfires.csv")
print(forest.head())
print(forest.describe())
print(forest.columns)

#select necessary column of the dataset.
f = forest.columns
forest1 = forest.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,30]]

#convert str into int.
m=forest1['month'].unique()
m1={}
for i in range(12):
    m1[i]=m[i]
m2=forest1['month']
for i in range(len(m2)):
    for j in range(12):
        if m2[i]==m1[j]:
            m2[i]=j

d=forest1['day'].unique()
d1={}
for i in range(7):
    d1[i]=d[i]
d2=forest1['day']
for i in range(len(d2)):
    for j in range(7):
        if d2[i]==d1[j]:
            d2[i]=j

s=forest1['size_category'].unique()
s1={}
for i in range(2):
    s1[i]=s[i]
s2=forest1['size_category']
for i in range(len(s2)):
    for j in range(2):
        if s2[i]==s1[j]:
            s2[i]=j

#by using for condition print boxplot between
#  size_category vs FFMC,DMC,DC,ISI,temp,wind,rain.area
arr=forest1.columns[2:10]
for i in arr:
    y=i
    sns.boxplot(x="size_category",y=y,data=forest1,palette = "hls")
    plt.show()

#show distplot.
sns.distplot(forest1['RH'])
plt.show()
sns.distplot(forest1['temp'])
plt.show()
sns.distplot(forest1['ISI'])
plt.show()
sns.distplot(forest1['DC'])
plt.show()
sns.distplot(forest1['FFMC'])
plt.show()
sns.distplot(forest1['DMC'])
plt.show()

#plot bar grap of size_category.
size = forest1['size_category'].value_counts()
for i, v in enumerate(size):
    plt.text(i,
              v,
              size[i],
              fontsize=15,
              color="red")
plt.hist(forest1['size_category'],color='green',edgecolor='black')
plt.xlabel("Size_category")
plt.show()

sns.pairplot(data=forest1)
plt.show()

#split dataset into train and test
y = forest1['size_category']
x = forest1.drop(['size_category'],axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#convert into str type.
Y_train = Y_train.astype("int")
Y_test = Y_test.astype("int")

# Create SVM classification object
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(X_train,Y_train)
pred_test_linear = model_linear.predict(X_test)
np.mean(pred_test_linear==Y_test)

from sklearn.metrics import classification_report

print("linear Classification Report \n",classification_report(Y_test,pred_test_linear),'\n\n\n')

from sklearn.metrics import confusion_matrix

print("linear Confusion Matrix \n",confusion_matrix(Y_test,pred_test_linear),'\n\n\n')

from sklearn.metrics import accuracy_score

print("linear Accuracy Score : ",accuracy_score(Y_test,pred_test_linear),'\n\n\n')

#kernal = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(X_train,Y_train)
pred_test_poly = model_poly.predict(X_test)
np.mean(pred_test_poly==Y_test)

from sklearn.metrics import classification_report

print("Poly Classification Report \n",classification_report(Y_test,pred_test_poly),'\n\n\n')

from sklearn.metrics import confusion_matrix

print("Poly Confusion Matrix \n",confusion_matrix(Y_test,pred_test_poly),'\n\n\n')

from sklearn.metrics import accuracy_score

print("Poly Accuracy Score : ",accuracy_score(Y_test,pred_test_poly),'\n\n\n')

#kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(X_train,Y_train)
pred_test_rbf = model_rbf.predict(X_test)
np.mean(pred_test_rbf==Y_test)

from sklearn.metrics import classification_report

print("RBF Classification Report \n",classification_report(Y_test,pred_test_rbf),'\n\n\n')

from sklearn.metrics import confusion_matrix

print("RBF Confusion Matrix \n",confusion_matrix(Y_test,pred_test_rbf),'\n\n\n')

from sklearn.metrics import accuracy_score

print("RBF Accuracy Score : ",accuracy_score(Y_test,pred_test_rbf),'\n\n\n')



