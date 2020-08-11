##### Import pandas, numpy, matplotlib,and seaborn. Even I imported sklearn when required.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

############################ Check out the Data ########################################

# We'll be working with this csv file from the assignment.
#
dataset_1 = [line.rstrip() for line in open('/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/6. Logistic Regression - Done/bank-full.csv')]
print(type(dataset_1))
print(len(dataset_1))
print(dataset_1[0].split('";"'))

dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/6. Logistic Regression - Done/bank-full.csv", sep=';',names=["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"])
dataset.drop(0,inplace=True)
dataset.columns.names = ['Index']

# dataset.drop(dataset.index[[9]],axis=0,inplace=True)

######## Change in data Type ################

dataset["age"]= dataset["age"].astype(int)
dataset["job"]= dataset["job"].astype(str)
dataset["marital"]= dataset["marital"].astype(str)
dataset["education"]= dataset["education"].astype(str)
dataset["default"]= dataset["default"].astype(str)
dataset["balance"]= dataset["balance"].astype(int)
dataset["housing"]= dataset["housing"].astype(str)
dataset["loan"]= dataset["loan"].astype(str)
dataset["contact"]= dataset["contact"].astype(str)
dataset["day"]= dataset["day"].astype(int)
dataset["month"]= dataset["month"].astype(str)
dataset["duration"]= dataset["duration"].astype(int)
dataset["campaign"]= dataset["campaign"].astype(int)
dataset["pdays"]= dataset["pdays"].astype(int)
dataset["previous"]= dataset["previous"].astype(int)
dataset["poutcome"]= dataset["poutcome"].astype(str)
dataset["y"]= dataset["y"].astype(str)

######## Basic Analysis ################

print(dataset.head(),'\n')

print(dataset.info(),'\n')

print(dataset.describe(),'\n')

print(dataset.columns,'\n')


############################ Creating a Dummy Variable ##############################

job1 = pd.get_dummies(dataset['job'],drop_first=True)
marital1 = pd.get_dummies(dataset['marital'],drop_first=True)
education1 = pd.get_dummies(dataset['education'],drop_first=True)
default1 = pd.get_dummies(dataset['default'],drop_first=True)
default1 = default1.rename(columns={'yes':'Dyes'})
housing1 = pd.get_dummies(dataset['housing'],drop_first=True)
housing1 =housing1.rename(columns={'yes':'Hyes'})
loan1 = pd.get_dummies(dataset['loan'],drop_first=True)
loan1 = loan1.rename(columns={'yes':'lyes'})
contact1 = pd.get_dummies(dataset['contact'],drop_first=True)
month1 = pd.get_dummies(dataset['month'],drop_first=True)
poutcome1 = pd.get_dummies(dataset['poutcome'],drop_first=True)
y1 = pd.get_dummies(dataset['y'],drop_first=True)
y1 = y1.rename(columns={'yes':'Yyes'})

############################ Concating the dummy Columns ##############################

dataset_2 = pd.concat([dataset,job1,marital1,education1,default1,housing1,loan1,contact1,month1,poutcome1,y1],axis=1)
print(dataset_2.head())

########################### Exploratory Data Analysis (EDA) #########################
########################### creating some simple plots to check out the data! #########################


######## EDA - Graphs ##########

plt.figure(figsize=(10,7))
sns.countplot(x='y',data=dataset_2)
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(x="y",y='age',data= dataset_2) ## X has to be catagorical or Discreat
plt.show()

sns.jointplot(data = dataset_2, x = 'campaign', y='balance') ## X & Y has to be continous
plt.show()

sns.jointplot(x = 'balance', y='age', kind='hex',data=dataset_2)
plt.show()

sns.pairplot(dataset,height= 2)
plt.show()

sns.lmplot(x = 'balance', y='age',data=dataset)
plt.show()

sns.pairplot(dataset,hue='y')
plt.show()

############################# Training and Testing data ##########################
# ######## Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable,
# ######## In this case the Weight Column.

######## X and y arrays ##########

dataset_2.drop(['job', 'marital', 'education', 'default','housing','loan', 'contact', 'day', 'month', 'poutcome', 'y'],axis=1,inplace=True)
x = dataset_2.drop('Yyes',axis=1)
y = dataset_2['Yyes']

#   ######## Train Test Split ##########

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=3)


#############_______________________________________________________________________________________________________________###############
############# Type 2 --> Logistic Regression
#############_______________________________________________________________________________________________________________###############

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)

#   ######## Model Evaluation ##########
#   ######## Evaluating the model by checking out it's coefficients and how we can interpret them.##########

# ###########################  Predictions from our Model ##########################
#   ######## Grabing predictions off our test set and see how well it did! ##########

predictions = logmodel.predict(x_test)

print(predictions,'\n')

print(y_test,'\n')

###############################  Evaluating the Model   ###########################

#  ######## Option 2 ------- Logistic Regression ##########
############### We will check precision,recall,f1-score using classification report! #############

from sklearn.metrics import classification_report

print("classification_report \n",classification_report(y_test,predictions),'\n\n\n')

from sklearn.metrics import confusion_matrix

print("confusion_matrix \n",confusion_matrix(y_test,predictions),'\n\n\n')

from sklearn.metrics import accuracy_score

print("accuracy_score : ",accuracy_score(y_test,predictions))