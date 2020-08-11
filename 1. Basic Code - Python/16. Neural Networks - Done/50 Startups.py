# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda
from keras.utils import plot_model
from ann_visualizer.visualize import ann_viz
# print dataset
startup = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/16. Neural Networks - Done/50_Startups.csv")
print(startup.head(),'\n\n\n')

# convert string into unique int value
s=startup['State'].unique()
s1={}
for i in range(3):
    s1[i]=s[i]
s2=startup['State']
for i in range(len(s2)):
    for j in range(3):
        if s2[i]==s1[j]:
            s2[i]=j

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

column_names = list(startup.columns)

# split dataset into predictor & target
predictors = column_names[0:4]
target = column_names[4]

# convert datatype into flot
ta = np.array(startup[target].astype('float'))
pu = np.array(startup[predictors].astype('float'))
print(ta.dtype,'\n\n\n')
print(pu.dtype,'\n\n\n')
print(startup.dtypes,'\n\n\n')

# print pred_train & rmse values of dataseet.
first_model = prep_model([4,50,1])
first_model.fit(pu,ta,epochs=5)
pred_train = first_model.predict(pu)
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-startup[target])**2))

# plot graph
plt.plot(pred_train,startup[target],"bo")
plt.xlabel('Pred_Train Vs Startup[Target]')
plt.show()

# print correlation value
print("\n\n",np.corrcoef(pred_train,startup[target]))

# neural network graph plot
ann_viz(first_model, view=True, title="Startup Neural Network")