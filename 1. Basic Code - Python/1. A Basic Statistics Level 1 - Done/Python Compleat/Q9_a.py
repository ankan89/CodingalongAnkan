# import packages
from typing import Any, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# print dataset
from pandas import Series, DataFrame
from pandas.io.parsers import TextFileReader

dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/1. A Basic Statistics Level 1 - Done/Q9_a.csv")
print(dataset.columns,'\n\n')
new_dataset=dataset.columns
print(new_dataset,'\n\n')

# print skewnesws & kurtosis
print('Skewness : \n',dataset.skew(),'\n\n')
print('Kurtosis : \n',dataset.kurt(),'\n\n')
print('Correlation : \n',dataset.corr(),'\n\n')
print('Dataset Head : \n',dataset.head(),'\n\n')

# print max & min, plot histogram, bar and categorise.
# Speed
print("Max speed : ",dataset["speed"].max(),'\n\n\n')
print("Min speed : ",dataset["speed"].min(),'\n\n\n')
plt.hist(dataset['speed'],edgecolor='k')
plt.xlabel('speed')
plt.show()

# Dist
print("Max dist : ",dataset["dist"].max(),'\n\n\n')
print("Min dist : ",dataset["dist"].min(),'\n\n\n')
plt.hist(dataset['dist'],edgecolor='k')
plt.xlabel('dist')
plt.show()


# print skewness,kurtosis and plot histogram, distplot, boxplot
# speed
print('Skewness speed : ',dataset['speed'].skew(),'\n\n\n')
print('Kurtosis speed : ',dataset['speed'].kurt(),'\n\n\n')
plt.hist(dataset['speed'],edgecolor='k')
plt.xlabel('speed')
plt.show()
sns.distplot(dataset['speed'],hist=False)
plt.xlabel('speed')
plt.show()
plt.boxplot(dataset['speed'])
plt.xlabel('speed')
plt.show()

# dist
print('Skewness dist : ',dataset['dist'].skew(),'\n\n\n')
print('Kurtosis dist : ',dataset['dist'].kurt(),'\n\n\n')
plt.hist(dataset['dist'],edgecolor='k')
plt.xlabel('dist')
plt.show()
sns.distplot(dataset['dist'],hist=False)
plt.xlabel('dist')
plt.show()
plt.boxplot(dataset['dist'])
plt.xlabel('dist')
plt.show()


# plot graph between speed and dist
df=pd.crosstab(dataset['speed'],dataset['dist'])
df.plot(kind="bar",stacked=True,)
plt.grid(axis="y")
plt.show()

# visulization
plt.plot(np.arange(50),dataset.speed)
plt.plot(np.arange(50),dataset.dist)
plt.show()
plt.plot(np.arange(50),dataset.speed,"ro-")
plt.plot(np.arange(50),dataset.dist,"ro-")
plt.show()
dataset.speed.value_counts().plot(kind="pie")
plt.show()
dataset.speed.value_counts().plot(kind="bar")
plt.show()
print('Corr speed : ',dataset.dist.corr(dataset.speed),'\n\n')

plt.plot(dataset.speed,dataset["dist"],"ro");plt.xlabel("speed");plt.ylabel("dist");plt.show()









































