import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/12. KNN - Done/glass.csv")

######## Basic Analysis ################

print(dataset.head(),"\n\n\n")

print(dataset.tail(),"\n\n\n")

print(dataset.info(),"\n\n\n")

print(dataset.describe(),"\n\n\n")

print(dataset.columns,"\n\n\n")

########################### Exploratory Data Analysis (EDA) #########################
########################### creating some simple plots to check out the data! #########################

plt.figure(figsize=(10,8))
sns.heatmap(dataset.corr(), annot=True,cmap ='RdYlGn')
plt.show()


####----------SubPlot--------####
f, axes = plt.subplots(1,2,figsize=(14,4))
####----------Plot 1--------####
sns.distplot(dataset['RI'], ax = axes[0])
axes[0].set_xlabel('Refractive Index', fontsize=14)
axes[0].set_ylabel('Count', fontsize=14)
axes[0].yaxis.tick_left()
####----------Plot 2--------####
sns.violinplot(x = 'Type', y = 'RI', data = dataset, hue = 'Type', dodge = False, ax = axes[1])
axes[1].set_xlabel('Type of glass', fontsize=14)
axes[1].set_ylabel('Refractive Index', fontsize=14)
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)

plt.show()


############## ----Data Standardization---- #########################

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dataset.drop('Type',axis=1))

############### Creating the feature to get a standardize dataframe #########################
scaled_feature = scaler.transform(dataset.drop('Type',axis=1))
std_dataset = pd.DataFrame(scaled_feature,columns=dataset.columns[:-1])
print(scaled_feature,"\n\n\n")
print(std_dataset.head(),"\n\n\n")

############################# Training and Testing data ##########################
# ######## Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable,
# ######## In this case the Weight Column.

######## X and y arrays ##########

x = std_dataset
y = dataset['Type']

  ####### Train Test Split ##########

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=42)

#############_______________________________________________________________________________________________________________###############
#############  KNN #####################
#############_______________________________________________________________________________________________________________###############

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)

# ###########################  Predictions from our Model ##########################
#   ######## Grabing predictions off our test set and see how well it did! ##########

predictions = knn.predict(x_test)
print(pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'], margins=True),"\n\n\n")
# print(predictions,"\n\n\n")


###############################  Evaluating the Model   ###########################


 ######## type 3 ------- KNN ##########
############# We will check precision,recall,f1-score using classification report! #############

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions),"\n\n\n")

from sklearn.metrics import confusion_matrix

# print(confusion_matrix(y_test,predictions),"\n\n\n")
#
from sklearn.metrics import accuracy_score
#
print(accuracy_score(y_test,predictions),"\n\n\n")

############# Determining the best K value and plotting it #############

error_rate = []

for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    predictions_i = knn.predict(x_test)
    error_rate.append(np.mean(predictions_i != y_test))

plt.figure(figsize=(10,6))
sns.lineplot(range(1,15),error_rate, color='blue', marker='o', linestyle='dashed',linewidth=2, markersize=12, markerfacecolor = 'red')
plt.title("Error Rate vs K Value")
plt.xlabel('Nos of Cluster')
plt.ylabel('Error Rate')
plt.show()

knn1 = KNeighborsClassifier(n_neighbors=3)
knn1.fit(x_train,y_train)
pred = knn1.predict(x_test)

print('confusion_matrix : \n',confusion_matrix(y_test,pred),"\n\n\n")
print('classification_report : \n',classification_report(y_test,pred),"\n\n\n")
print('accuracy_score : ',accuracy_score(y_test,pred),"\n\n\n")


# line plot through function
glass1 = []
for i in range(1, 20):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(x_train, y_train)
    pred_i = classifier.predict(x_test)
    glass1.append(np.mean(pred_i != y_test))
plt.plot(range(1,20), glass1, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.xlabel("Nos of Cluster")
plt.ylabel("Error Rate")
plt.show()

train_acc = np.mean(classifier.predict(x_train)==y_train)
test_acc = np.mean(classifier.predict(x_test)==y_test)
print('train_acc : ',train_acc,'\n')
print('test_acc : ',test_acc,'\n')


acc = []
for i in range(1,10,1):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(x_train, y_train)
    train_acc = np.mean(classifier.predict(x_train)==y_train)
    test_acc = np.mean(classifier.predict(x_test)==y_test)
    acc.append([train_acc,test_acc])

# Train accuracy plot
plt.plot(np.arange(1,10,1),[i[0] for i in acc],"bo-")
plt.xlabel("Nos of Cluster")
plt.ylabel("Train Accuracy")
plt.show()

# Test accuracy plot
plt.plot(np.arange(1,10,1),[i[1] for i in acc],"ro-")
plt.xlabel("Nos of Cluster")
plt.ylabel("Test Accuracy")
plt.show()

plt.plot(range(1,10,1), acc, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.xlabel("Nos of Cluster")
plt.ylabel("Test & Test Accuracy")
plt.show()

