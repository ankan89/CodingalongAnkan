##### Import pandas, numpy, matplotlib,and seaborn. Even I imported sklearn when required.

from typing import Any, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame
from pandas.io.parsers import TextFileReader

############################ Check out the Data ########################################

# We'll be working with this csv file from the assignment.
#

newdata: Union[Union[TextFileReader, Series, DataFrame, None], Any] = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/5. Multi Linear Regression - Done/50_Startups.csv")

dataset = newdata

dataset.drop(dataset.index[[46,48,49]],axis=0,inplace=True)

######## Basic Analysis ################

print(dataset.head(),'\n')

print(dataset.info(),'\n')

print(dataset.describe(),'\n')

print(dataset.columns,'\n')


########################### Exploratory Data Analysis (EDA) #########################
########################### creating some simple plots to check out the data! #########################

sns.jointplot(data = dataset, x = 'R&D Spend', y='Profit')
plt.show()

sns.jointplot(x = 'Marketing Spend', y='Profit', kind='hex',data=dataset)
plt.show()

sns.pairplot(dataset,height= 2)
plt.show()

sns.lmplot(x = 'Administration', y='Profit',data=dataset)
plt.show()

sns.distplot(dataset['Profit'])
plt.show()

sns.distplot(dataset['Marketing Spend'])
plt.show()

print(dataset.corr(),'\n')

sns.heatmap(dataset.corr(),annot=True)
plt.show()

############################ Training and Testing data ##########################
######## Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable,
######## In this case the Weight Column.

####### X and y arrays ##########

x = dataset[['R&D Spend', 'Administration', 'Marketing Spend']]
y = dataset['Profit']

# x = dataset.iloc[:,[0,1,2]]
# y = dataset.iloc[:,[4]]

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
    ######## Holding all other features fixed, a 1 unit increase in **R&D Spend ** ##########
    ######## Is associated with an **increase of (0.785423) **. ##########
    ######## Holding all other features fixed, a 1 unit increase in **Administration ** ##########
    ######## Is associated with an **increase of (-0.086787) **. ##########
    ######## Holding all other features fixed, a 1 unit increase in **Marketing Spend ** ##########
    ######## Is associated with an **increase of (0.026673) **. ##########


####### Option 2 ##########

import statsmodels.formula.api as smf

mlr1 = smf.ols('y~x',data=dataset).fit()

print("Model Summary : \n",mlr1.summary())

import statsmodels.api as sm
sm.graphics.influence_plot(mlr1)
plt.show()


########################  Predictions from our Model ##########################
  ##### Grabing predictions off our test set and see how well it did! ##########

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



###############################  Creating a Table containing R^2 Values   ###########################

print('Model_1 : 0.90')
print('Model_2 : 0.93')
print('Model_3 : 0.96')
