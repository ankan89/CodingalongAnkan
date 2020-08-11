# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from ann_visualizer.visualize import ann_viz

# print dataset
forestfire = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/16. Neural Networks - Done/forestfires.csv")
forestfire.head()

# convert string into unique int values
# month
m=forestfire['month'].unique()
m1={}
for i in range(12):
    m1[i]=m[i]
m2=forestfire['month']
for i in range(len(m2)):
    for j in range(12):
        if m2[i]==m1[j]:
            m2[i]=j
# day
d=forestfire['day'].unique()
d1={}
for i in range(7):
    d1[i]=d[i]
d2=forestfire['day']
for i in range(len(d2)):
    for j in range(7):
        if d2[i]==d1[j]:
            d2[i]=j
# size category
s=forestfire['size_category'].unique()
s1={}
for i in range(2):
    s1[i]=s[i]
s2=forestfire['size_category']
for i in range(len(s2)):
    for j in range(2):
        if s2[i]==s1[j]:
            s2[i]=j

# select column
f = forestfire.columns
forest = forestfire.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,30]]

# function for neural network
def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

# devided dataset inti predictor & target
column_names = list(forest.columns)
predictors = forest.iloc[:,[0,1,2,3,4,5,6,7,8,9,11]]
target = forest.iloc[:,10]
target =target.astype('int')
pu=np.array(predictors).astype('int')

# print pred_train & rmse values of dataseet.
first_model = prep_model([11,517,1])
first_model.fit(pu,np.array(target),epochs=9)
pred_train = first_model.predict(pu)
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-target)**2))

# plot graph
plt.plot(pred_train,target,"bo")
plt.xlabel("Predict Train")
plt.ylabel("Target")
plt.show()

# neural network graph plot
ann_viz(first_model, title="forestfire neural network")
