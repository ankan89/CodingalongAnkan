# import pacakages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

# print dataset.
dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/13. Decision Trees - Done/Company_Data.csv")
dataset.head()
colnames = list(dataset.columns)

# Convert string into unique integer values.
dataset.loc[dataset['ShelveLoc']=='Good','ShelveLoc']=0
dataset.loc[dataset['ShelveLoc']=='Bad','ShelveLoc']=1
dataset.loc[dataset['ShelveLoc']=='Medium','ShelveLoc']=2

dataset.loc[dataset['Urban']=='Yes','Urban']=1
dataset.loc[dataset['Urban']=='No','Urban']=0

dataset.loc[dataset['US']=='Yes','US']=1
dataset.loc[dataset['US']=='No','US']=0

# plot histogram
plt.hist(dataset['Urban'],edgecolor='k')
plt.grid(axis='y')
plt.xlabel('Urban')
plt.show()

# check skewnee & kurtosis
# graph plot
dataset['Sales'].skew()
dataset['Sales'].kurt()
plt.hist(dataset['Sales'],edgecolor='k')
plt.xlabel("Sales")
plt.show()
sns.distplot(dataset['Sales'],hist=False)
plt.xlabel("Sales")
plt.show()
plt.boxplot(dataset['Sales'])
plt.xlabel("Sales")
plt.show()

dataset['CompPrice'].skew()
dataset['CompPrice'].kurt()
plt.hist(dataset['CompPrice'],edgecolor='k')
plt.xlabel("CompPrice")
plt.show()
sns.distplot(dataset['CompPrice'],hist=False)
plt.xlabel("CompPrice")
plt.show()
plt.boxplot(dataset['CompPrice'])
plt.xlabel("CompPrice")
plt.show()

# split train & test dataset
x = dataset.drop(['Sales'],axis=1)
y = dataset['Sales']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# implement linear regression and plot graph.
dataset1 = LinearRegression()
dataset1.fit(X_train,Y_train)
pred = dataset1.predict(X_test)
dataset1.score(X_train,Y_train)

x_l = range(len(X_test))
plt.scatter(x_l, Y_test, s=5, color="green", label="Test Original")
plt.plot(x_l, pred, lw=0.8, color="black", label="Test Predicted")
plt.legend()
plt.show()

# implement rigid regression & plt scatter graph.
alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,0.5, 1]
for a in alphas:
   model = Ridge(alpha=a, normalize=True).fit(X_train,Y_train)
   score = model.score(X_train,Y_train)
   pred_y = model.predict(X_test)
   mse = mean_squared_error(Y_test, pred_y)
   print("Alpha:{0:.6f}, R2:{1:.3f}, MSE:{2:.2f}, RMSE:{3:.2f}"
    .format(a, score, mse, np.sqrt(mse)))

# print ypred, score, mse
ridge_mod=Ridge(alpha=0.01, normalize=True).fit(X_train,Y_train)
ypred = ridge_mod.predict(X_test)
score = model.score(X_test,Y_test)
mse = mean_squared_error(Y_test,ypred)
print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
   .format(score, mse,np.sqrt(mse)))

# plot scatter diagram
x_ax = range(len(X_test))
plt.scatter(x_ax, Y_test, s=5, color="blue", label="Test Original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="Test Predicted")
plt.legend()
plt.show()
