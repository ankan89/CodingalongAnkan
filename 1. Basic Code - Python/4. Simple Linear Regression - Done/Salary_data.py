##### Import pandas, numpy, matplotlib,and seaborn. Even I imported sklearn when required.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get the Data
# We'll be working with the .csv file from the assignment. It has the info of 'YearsExperience' and 'Salary'.
#
dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/4. Simple Linear Regression - Done/Salary_Data.csv")

# dataset.drop(dataset.index[[19,28,29]],axis=0,inplace=True) ##### Not required

######## Basic Analysis ################

print(dataset.head(),'\n')

print(dataset.info(),'\n')

print(dataset.describe(),'\n')

print(dataset.columns,'\n')

######################### Exploratory Data Analysis (EDA) #########################

sns.jointplot(data = dataset, x = 'YearsExperience', y='Salary')
plt.show()

sns.jointplot(x = 'YearsExperience', y='Salary', kind='hex',data=dataset)
plt.show()

sns.pairplot(dataset,height= 2)
plt.show()

sns.lmplot(x = 'YearsExperience', y='Salary',data=dataset)
plt.show()

sns.distplot(dataset['Salary'])
plt.show()

sns.distplot(dataset['YearsExperience'])
plt.show()

print(dataset.corr(),'\n')

sns.heatmap(dataset.corr(),annot=True)
plt.show()

############################ Training and Testing data ##########################
######## Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable,
######## In this case the Weight Column.

####### X and y arrays ##########

x = dataset[['YearsExperience']]
y = dataset['Salary']

  ######## Option 1 ##########

  ######## Train Test Split ##########

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,random_state=3)

  ######## Creating and Training the Mode ##########

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(x_train,y_train)

  ######## Model Evaluation ##########
  ######## Evaluating the model by checking out it's coefficients and how we can interpret them.##########

print(lm.intercept_,'\n')

print(lm.coef_,'\n')

print(x_train.columns,'\n')

cdf = pd.DataFrame(lm.coef_,x.columns,columns=['Coeff'])

print(cdf)

   ######## Interpreting the coefficients: ##########
   ######## Holding all other features fixed, a 1 unit increase in **YearsExperience** ##########
   ######## Is associated with an **increase of (9594.849434) **. ##########


####### Option 2 ##########

import statsmodels.formula.api as smf

mlr1 = smf.ols('y~x',data=dataset).fit()

print(mlr1.summary())

import statsmodels.api as sm
sm.graphics.influence_plot(mlr1)
plt.show()

# newdata1 = dataset.drop(dataset.index[[20,4]],axis=0)
# x1 = newdata1[['YearsExperience']]
# y1 = newdata1['Salary']
#
# mlr2 = smf.ols('y1~x1',data=newdata1).fit()
#
# print(mlr2.summary())
#
# sm.graphics.influence_plot(mlr2)
# plt.show()


###########################  Predictions from our Model ##########################
  ######## Grabing predictions off our test set and see how well it did! ##########

predictions = lm.predict(x_test)

print(predictions,'\n')

print(y_test,'\n')

plt.scatter(y_test,predictions)
plt.xlabel("Y Test(TrueValue)")
plt.show()

#############################   Residual   ######################################
  ######## Residual Histogram ##########

sns.distplot((y_test-predictions))
plt.show()

##############################  Evaluating the Model   ###########################
######## Regression Evaluation Metrics ##########
######## Here are three common evaluation metrics for regression problems:##########

######## Mean Absolute Error (MAE) is the mean of the absolute value of the errors ##########
######## Mean Squared Error (MSE) is the mean of the squared errors:##########
######## Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:##########

######## Comparing the above metrics:##########
######## MAE is the easiest to understand, because it's the average error.##########
######## MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.##########
######## RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.##########

######## All of the above are loss functions which has to be minimize for better performance.##########

from sklearn import metrics

print('MAE :',metrics.mean_absolute_error(y_test,predictions),'\n')

print('MSE :',metrics.mean_squared_error(y_test,predictions),'\n')

print('RMSE :',np.sqrt(metrics.mean_squared_error(y_test,predictions)),'\n')

print('Var(R^2) :',metrics.explained_variance_score(y_test,predictions))

