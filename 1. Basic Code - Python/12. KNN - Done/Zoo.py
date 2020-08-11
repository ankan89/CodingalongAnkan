import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/12. KNN - Done/Zoo.csv")

######## Basic Analysis ################

print(dataset.head(),"\n\n\n")

print(dataset.tail(),"\n\n\n")

print(dataset.info(),"\n\n\n")

print(dataset.describe(),"\n\n\n")

print(dataset.columns,"\n\n\n")

print(dataset.shape,"\n\n\n")

print(dataset.iloc[:,:],"\n\n\n")


########################### Exploratory Data Analysis (EDA) #########################
########################### creating some simple plots to check out the data! #########################

plt.figure(figsize=(10,8))
sns.heatmap(dataset.corr(), annot=True,cmap ='RdYlGn')
plt.show()


############### ----Data Standardization---- #########################

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(dataset.iloc[:,1:18])

############### Creating the feature to get a standardize dataframe #########################
scaled_feature = scaler.transform(dataset.iloc[:,1:18])
print(pd.DataFrame(scaled_feature))
std_dataset = pd.DataFrame(scaled_feature, columns=dataset.columns[1:18])
print(scaled_feature, "\n\n\n")
print(std_dataset.head(), "\n\n\n")

############################## Training and Testing data ##########################
# ######## Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable,
# ######## In this case the Weight Column.

######## X and y arrays ##########

x = std_dataset.iloc[:,0:16]
print("X - Label ---",x.columns,"\n\n\n")
y = dataset.iloc[:,17:18]
print("Y - Label ---",y.columns,"\n\n\n")


######## Train Test Split ##########

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print('Train Test Split done')

#############_______________________________________________________________________________________________________________###############
#############  KNN #####################
#############_______________________________________________________________________________________________________________###############

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train.values.ravel())

print("Model Trained")

# ###########################  Predictions from our Model ##########################
#   ######## Grabing predictions off our test set and see how well it did! ##########

predictions = knn.predict(x_test)
print(pd.crosstab(y_test.values.ravel(), predictions, rownames=['True'], colnames=['Predicted'], margins=True), "\n\n\n")
# print(predictions,"\n\n\n")


###############################  Evaluating the Model   ###########################


######## type 3 ------- KNN ##########
############# We will check precision,recall,f1-score using classification report! #############

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions), "\n\n\n")

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,predictions),"\n\n\n")

from sklearn.metrics import accuracy_score

#
print(accuracy_score(y_test, predictions), "\n\n\n")

############# Determining the best K value and plotting it #############

error_rate = []

for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train.values.ravel())
    predictions_i = knn.predict(x_test)
    error_rate.append(np.mean(predictions_i != y_test.values.ravel()))

plt.figure(figsize=(10, 6))
sns.lineplot(range(1, 15), error_rate, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=12,
             markerfacecolor='red')
plt.title("Error Rate vs K Value")
plt.xlabel('Nos of Cluster')
plt.ylabel('Error Rate')
plt.show()

knn1 = KNeighborsClassifier(n_neighbors=7)
knn1.fit(x_train, y_train.values.ravel())
pred = knn1.predict(x_test)

print(pd.crosstab(y_test.values.ravel(), pred, rownames=['True'], colnames=['Predicted'], margins=True), "\n\n\n")
print('classification_report : \n',classification_report(y_test, pred), "\n\n\n")
print('accuracy_score : ',accuracy_score(y_test, pred), "\n\n\n")

print("As the error starts increasing with the increase in K value so K value @ 1 is the best predictor")




# line plot through function
animal_error = []
for i in range(1, 20):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(x_train, y_train.values.ravel())
    pred_i = classifier.predict(x_test)
    animal_error.append(np.mean(pred_i != y_test.values.ravel()))
plt.plot(range(1,20), animal_error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.xlabel("Nos of Cluster")
plt.ylabel("Error Rate")
plt.show()

train_acc = np.mean(classifier.predict(x_train)==y_train.values.ravel())
test_acc = np.mean(classifier.predict(x_test)==y_test.values.ravel())
print('train_acc : ',train_acc,'\n')
print('test_acc : ',test_acc,'\n')


acc = []
for i in range(1,10,1):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(x_train, y_train.values.ravel())
    train_acc = np.mean(classifier.predict(x_train)==y_train.values.ravel())
    test_acc = np.mean(classifier.predict(x_test)==y_test.values.ravel())
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