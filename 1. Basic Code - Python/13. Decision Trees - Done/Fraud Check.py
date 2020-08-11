# import pacakages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
import seaborn as sns
# print dataset.
dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/13. Decision Trees - Done/Fraud_check.csv")
dataset.head()
colnames = list(dataset.columns)
# Convert string into unique integer values.
dataset.loc[dataset['Undergrad']=='YES','Undergrad']=1
dataset.loc[dataset['Undergrad']=='NO','Undergrad']=0

dataset.loc[dataset['Marital.Status']=='Single','Marital.Status']=1
dataset.loc[dataset['Marital.Status']=='Married','Marital.Status']=2
dataset.loc[dataset['Marital.Status']=='Divorced','Marital.Status']=3

dataset.loc[dataset['Urban']=='YES','Urban']=1
dataset.loc[dataset['Urban']=='NO','Urban']=0

dataset.loc[dataset['Taxable.Income']<=30000,'Taxable.Income']=1
dataset.loc[dataset['Taxable.Income']>30000,'Taxable.Income']=0
# plot histogram
plt.hist(dataset['Urban'],edgecolor='k')
plt.grid(axis='y')
plt.show()

# check skewnee & kurtosis
# graph plot
dataset['City.Population'].skew()
dataset['City.Population'].kurt()
plt.hist(dataset['City.Population'],edgecolor='k')
sns.distplot(dataset['City.Population'],hist=False)
plt.boxplot(dataset['City.Population'])

dataset['Work.Experience'].skew()
dataset['Work.Experience'].kurt()
plt.hist(dataset['Work.Experience'],edgecolor='k')
sns.distplot(dataset['Work.Experience'],hist=False)
plt.boxplot(dataset['Work.Experience'])

# count values
dataset['Taxable.Income'].value_counts()

# split train & test dataset
y = dataset['Taxable.Income']
x = dataset.drop(['Taxable.Income'],axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# implement decission tree classifier
# criterion = 'entropy',class_weight='balanced'
fr1 = DecisionTreeClassifier(criterion = 'entropy',class_weight='balanced',)
fr1.fit(X_train,Y_train)
pred = fr1.predict(X_test)
print("Accuracy Score",accuracy_score(Y_test,pred),'\n')
print("Confusion Matrix : \n",confusion_matrix(Y_test,pred,labels=[1,0]),'\n\n\n')

# criterion = 'gini',class_weight='balanced',splitter='random',max_features='int'
fr2 = DecisionTreeClassifier(criterion = 'gini',class_weight='balanced',splitter='random',max_features="auto")
fr2.fit(X_train,Y_train)
pred_1 = fr2.predict(X_test)
print("Accuracy Score",accuracy_score(Y_test,pred_1),'\n')
print("Confusion Matrix : \n",confusion_matrix(Y_test,pred_1,labels=[1,0]),'\n\n\n')

# Bagging classifier
# max_samples=0.5,max_features=1.0,n_estimators=10
fr4 = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=10)
fr4.fit(X_train,Y_train)
pred_2 = fr4.predict(X_test)
print("Accuracy Score",accuracy_score(Y_test,pred_2),'\n')
print("Confusion Matrix : \n",confusion_matrix(Y_test,pred_2,labels=[1,0]),'\n\n\n')

# max_samples=0.5,max_features=1.0,n_estimators=10,random_state='int'
fr5 = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=10,random_state=42)
fr5.fit(X_train,Y_train)
pred_3 = fr5.predict(X_test)
print("Accuracy Score",accuracy_score(Y_test,pred_3),'\n')
print("Confusion Matrix : \n",confusion_matrix(Y_test,pred_3,labels=[1,0]),'\n\n\n')

