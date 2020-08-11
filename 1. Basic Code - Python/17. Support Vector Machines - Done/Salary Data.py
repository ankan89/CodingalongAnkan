# import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# print dataset
sal_train = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/17. Support Vector Machines - Done/SalaryData_Test(1).csv")
sal_test = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/17. Support Vector Machines - Done/SalaryData_Train(1).csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native","Salary"]

# plot boxplot
sns.boxplot(x="age",y="Salary",data=sal_train,palette = "hls")
plt.show()
sns.boxplot(x="age",y="Salary",data=sal_test,palette = "hls")
plt.show()

# function for fitting test & train data
number = preprocessing.LabelEncoder()
for i in string_columns:
    sal_train[i] = number.fit_transform(sal_train[i])
    sal_test[i] = number.fit_transform(sal_test[i])

# split dataset into trainx,y & testx,y
colnames = sal_train.columns
len(colnames[0:13])
trainX = sal_train[colnames[0:13]]
trainY = sal_train[colnames[13]]
testX  = sal_test[colnames[0:13]]
testY  = sal_test[colnames[13]]

# apply poly kernel
model_poly = SVC(kernel = "poly")
model_poly.fit(trainX,trainY)
pred_test_poly = model_poly.predict(testX)
np.mean(pred_test_poly==testY)

from sklearn.metrics import classification_report

print("Poly Classification Report \n",classification_report(testY,pred_test_poly),'\n\n\n')

from sklearn.metrics import confusion_matrix

print("Poly Confusion Matrix \n",confusion_matrix(testY,pred_test_poly),'\n\n\n')

from sklearn.metrics import accuracy_score

print("Poly Accuracy Score : ",accuracy_score(testY,pred_test_poly),'\n\n\n')

# apply rbf kernel
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(trainX,trainY)
pred_test_rbf = model_rbf.predict(testX)
np.mean(pred_test_rbf==testY)

from sklearn.metrics import classification_report

print("RBF Classification Report \n",classification_report(testY,pred_test_rbf),'\n\n\n')

from sklearn.metrics import confusion_matrix

print("RBF Confusion Matrix \n",confusion_matrix(testY,pred_test_rbf),'\n\n\n')

from sklearn.metrics import accuracy_score

print("RBF Accuracy Score : ",accuracy_score(testY,pred_test_rbf),'\n\n\n')
