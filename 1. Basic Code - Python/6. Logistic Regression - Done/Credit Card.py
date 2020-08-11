# import pacakegs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#print dataset
credit = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/6. Logistic Regression - Done/creditcard.csv")
print(credit.head(10),'\n\n')
print(credit.columns,'\n\n')

#count , plot bar graph & convert string into unique integer vaalues
# card column
credit.loc[credit['card']=='yes','card']=1
credit.loc[credit['card']=='no','card']=0
c1 = [i for i in credit.card if i==0 ]
c2 = [i for i in credit.card if i==1 ]
c3 = [len(c1),len(c2)]
c4 = ["0","1"]
for i, v in enumerate(c3):
    plt.text(i-.25,
              v,
              c3[i],
              fontsize=18,
              color="red")
plt.bar(c4,c3)
plt.xticks(c4,rotation=0)
plt.xlabel('Card')
plt.show()

# owner column
credit.loc[credit['owner']=='yes','owner']=1
credit.loc[credit['owner']=='no','owner']=0
o1 = [i for i in credit.owner if i==0 ]
o2 = [i for i in credit.owner if i==1 ]
o3 = [len(o1),len(o2)]
o4 = ["0","1"]
for i, v in enumerate(o3):
    plt.text(i-.25,
              v,
              o3[i],
              fontsize=18,
              color="red")
plt.bar(o4,o3)
plt.xticks(o4,rotation=0)
plt.xlabel('Owner')
plt.show()

# selfemo column
credit.loc[credit['selfemp']=='yes','selfemp']=1
credit.loc[credit['selfemp']=='no','selfemp']=0
s1 = [i for i in credit.selfemp if i==0 ]
s2 = [i for i in credit.selfemp if i==1 ]
s3 = [len(s1),len(s2)]
s4 = ["0","1"]
for i, v in enumerate(s3):
    plt.text(i-.25,
              v,
              s3[i],
              fontsize=18,
              color="red")
plt.bar(s4,s3)
plt.xticks(s4,rotation=0)
plt.xlabel('Selfemp')
plt.show()

# scatter plot between income & expenditure
plt.scatter(credit.income,credit.expenditure)
plt.xlabel('income')
plt.ylabel('expenditure')
plt.title('income vs expenditure')
plt.grid(True)
plt.show()

# scatter plt between age & expenditure
plt.scatter(credit.age,credit.expenditure)
plt.xlabel('age')
plt.ylabel('expenditure')
plt.title('age vs expenditure')
plt.grid(True)
plt.show()

#scatter plot between age & income
plt.scatter(credit.age,credit.income)
plt.xlabel('age')
plt.ylabel('income')
plt.title('age vs income')
plt.grid(True)
plt.show()

# scatter plot between income & share
plt.scatter(credit.income,credit.share)
plt.xlabel('income')
plt.ylabel('share')
plt.title('income vs share')
plt.grid(True)
plt.show()

# split into train and test dataset
credit.shape
X = credit.iloc[:,2:12]
Y = credit.iloc[:,1].astype('int')
xtrain,xtest,ytrain,ytest = train_test_split(X,Y)

# Logistic regression
classifier = LogisticRegression()
classifier.fit(xtrain,ytrain)
y_pred = classifier.predict(xtest)   # value predict
print("accuracy_score : ",accuracy_score(y_pred,ytest),'\n\n')            # calculate accuracy
print("f1_score : ",f1_score(y_pred,ytest),'\n\n')               # calculate f1 score
print("precision_score : ",precision_score(ytest, y_pred),'\n\n')       # calculate precision score
print("recall_score : ",recall_score(ytest,y_pred),'\n\n')             # calculate recall score


from sklearn.metrics import confusion_matrix

print("confusion_matrix \n",confusion_matrix(ytest,y_pred),'\n\n\n')

# predict probabilities
ns_probs = [0 for _ in range(len(ytest))]
lr_probs = classifier.predict_proba(xtest)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(ytest, ns_probs)
lr_auc = roc_auc_score(ytest, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(ytest, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')     # axis labels
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

































































