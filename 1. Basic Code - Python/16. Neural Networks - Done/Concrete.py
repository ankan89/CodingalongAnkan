# import pacakages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from ann_visualizer.visualize import ann_viz
import seaborn as sns

# print dataset
Concrete = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/Neural Networks/concrete.csv")
Concrete.head()

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

# split dataset in predictors & target
column_names = list(Concrete.columns)
predictors = column_names[0:8]
target = column_names[8]

# print pred_train & rmse values of dataseet.
first_model = prep_model([8,50,1])
first_model.fit(np.array(Concrete[predictors]),np.array(Concrete[target]),epochs=9)
pred_train = first_model.predict(np.array(Concrete[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-Concrete[target])**2))

# print correlation coefficient
np.corrcoef(pred_train,Concrete[target])

# neural network graph plot
ann_viz(first_model, view=True, title="Concrete Neural Network")

# plot graph for visualization.
plt.hist(Concrete['age'],edgecolor='k')
plt.xlabel("Age")
plt.show()
sns.distplot(Concrete['age'],hist=False)
plt.show()
plt.plot(pred_train,Concrete[target],"bo")
plt.xlabel("Predict Train")
plt.ylabel("Concrete [target]")
plt.show()
plt.scatter(Concrete['cement'],y=Concrete['fineagg'],color='green',alpha=.6)
plt.xlabel("Cement")
plt.ylabel("Fineagg")
plt.show()
plt.scatter(Concrete['cement'],y=Concrete['water'],color='red',alpha=.6)
plt.xlabel("Cement")
plt.ylabel("Water")
plt.show()
