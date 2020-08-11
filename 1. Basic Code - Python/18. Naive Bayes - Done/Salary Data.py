# import dataset
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# print dataset
salarytrain = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/18. Naive Bayes - Done/SalaryData_Test.csv")
salarytrain.columns
# convert string into unoque integer values
# workclass
w=salarytrain['workclass'].unique()
w1={}
for i in range(7):
    w1[i]=w[i]
w2=salarytrain['workclass']
for i in range(len(w2)):
    for j in range(7):
        if w2[i]==w1[j]:
            w2[i]=j
# education
e=salarytrain['education'].unique()
e1={}
for i in range(16):
    e1[i]=e[i]
e2=salarytrain['education']
for i in range(len(e2)):
    for j in range(16):
        if e2[i]==e1[j]:
            e2[i]=j
# maritalstatus
m=salarytrain['maritalstatus'].unique()
m1={}
for i in range(7):
    m1[i]=m[i]
m2=salarytrain['maritalstatus']
for i in range(len(m2)):
    for j in range(7):
        if m2[i]==m1[j]:
            m2[i]=j
# occupation
o=salarytrain['occupation'].unique()
o1={}
for i in range(14):
    o1[i]=o[i]
o2=salarytrain['occupation']
for i in range(len(o2)):
    for j in range(14):
        if o2[i]==o1[j]:
            o2[i]=j
# relationship
r=salarytrain['relationship'].unique()
r1={}
for i in range(6):
    r1[i]=r[i]
r2=salarytrain['relationship']
for i in range(len(r2)):
    for j in range(6):
        if r2[i]==r1[j]:
            r2[i]=j
# race
ra=salarytrain['race'].unique()
ra1={}
for i in range(5):
    ra1[i]=ra[i]
ra2=salarytrain['race']
for i in range(len(ra2)):
    for j in range(5):
        if ra2[i]==ra1[j]:
            ra2[i]=j
# sex
s=salarytrain['sex'].unique()
s1={}
for i in range(2):
    s1[i]=s[i]
s2=salarytrain['sex']
for i in range(len(s2)):
    for j in range(2):
        if s2[i]==s1[j]:
            s2[i]=j
# salary
n=salarytrain['Salary'].unique()
n1={}
for i in range(2):
    n1[i]=n[i]
n2=salarytrain['Salary']
for i in range(len(n2)):
    for j in range(2):
        if n2[i]==n1[j]:
            n2[i]=j
# native
na=salarytrain['native'].unique()
na1={}
for i in range(40):
    na1[i]=na[i]
na2=salarytrain['native']
for i in range(len(na2)):
    for j in range(40):
        if na2[i]==na1[j]:
            na2[i]=j
# drop column
salarytrain1 = salarytrain.drop(columns='educationno',axis=1)

# seperated input & output column
ip_columns = ["age","workclass","education","maritalstatus","occupation","relationship","race","sex","capitalgain","capitalloss","hoursperweek","native"]
op_column  = ["Salary"]

# split dataset into train & test
Xtrain,Xtest,ytrain,ytest = train_test_split(salarytrain1[ip_columns],salarytrain1['Salary'],test_size=0.3, random_state=0)

# apply gaussian & multinomial NB
ignb = GaussianNB()
imnb = MultinomialNB()

# convert into integer type
ytrain=ytrain.astype('int')
ytest=ytest.astype('int')

# fit Xtrain & Ytrain
ignb.fit(Xtrain,ytrain)
pred_gnb=ignb.predict(Xtest)

# predict value
pred_mnb = imnb.fit(Xtrain,ytrain).predict(Xtest)

# print confussion matrix
print("GaussianNB Confusion Matrix \n",confusion_matrix(ytest,pred_gnb),'\n\n\n')
pd.crosstab(ytest.values.flatten(),pred_gnb)
np.mean(pred_gnb==ytest.values.flatten())
print("GaussianNB Accuracy Score : ",accuracy_score(ytest,pred_gnb),'\n\n\n')

print("MultinomialNB Confusion Matrix \n",confusion_matrix(ytest,pred_mnb),'\n\n\n')
pd.crosstab(ytest.values.flatten(),pred_mnb)
np.mean(pred_mnb==ytest.values.flatten())
print("MultinomialNB Accuracy Score : ",accuracy_score(ytest,pred_mnb),'\n\n\n')
