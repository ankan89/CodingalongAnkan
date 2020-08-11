# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# print dataset
dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/1. A Basic Statistics Level 1 - Done/wc-at.csv")
print(dataset.columns,'\n\n')
new_dataset=dataset.columns
print(new_dataset,'\n\n')

# print skewnesws & kurtosis
print('Skewness : \n',dataset.skew(),'\n\n')
print('Kurtosis : \n',dataset.kurt(),'\n\n')
print('Correlation : \n',dataset.corr(),'\n\n')
print('Dataset Head : \n',dataset.head(),'\n\n')

# print max & min, plot histogram, bar and categorise.
# column waist
print("Max Waist : ",dataset["Waist"].max(),'\n\n\n')
print("Min Waist : ",dataset["Waist"].min(),'\n\n\n')
plt.hist(dataset['Waist'],edgecolor='k')
plt.xlabel('Waist')
plt.show()

# column AT
print("Max AT : ",dataset["AT"].max(),'\n\n\n')
print("Min AT : ",dataset["AT"].min(),'\n\n\n')
plt.hist(dataset['AT'],edgecolor='k')
plt.xlabel('AT')
plt.show()


# print skewness,kurtosis and plot histogram, distplot, boxplot
# Waist
print('Skewness Waist : ',dataset['Waist'].skew(),'\n\n\n')
print('Kurtosis Waist : ',dataset['Waist'].kurt(),'\n\n\n')
plt.hist(dataset['Waist'],edgecolor='k')
plt.xlabel('Waist')
plt.show()
sns.distplot(dataset['Waist'],hist=False)
plt.xlabel('Waist')
plt.show()
plt.boxplot(dataset['Waist'])
plt.xlabel('Waist')
plt.show()

# AT
print('Skewness AT : ',dataset['AT'].skew(),'\n\n\n')
print('Kurtosis AT : ',dataset['AT'].kurt(),'\n\n\n')
plt.hist(dataset['AT'],edgecolor='k')
plt.xlabel('AT')
plt.show()
sns.distplot(dataset['AT'],hist=False)
plt.xlabel('AT')
plt.show()
plt.boxplot(dataset['AT'])
plt.xlabel('AT')
plt.show()

# plot graph between Waist and AT
df=pd.crosstab(dataset['Waist'],dataset['AT'])
df.plot(kind="line",stacked=True,)
plt.grid(axis="y")
plt.show()

# visulization
plt.plot(np.arange(109),dataset.Waist)
plt.plot(np.arange(109),dataset.AT)
plt.show()
plt.plot(np.arange(109),dataset.Waist,"ro-")
plt.plot(np.arange(109),dataset.AT,"ro-")
plt.show()
dataset.Waist.value_counts().plot(kind="pie")
plt.show()
dataset.Waist.value_counts().plot(kind="bar")
plt.show()
print('Corr Waist',dataset.Waist.corr(dataset.Waist),'\n\n')
print('Corr AT',dataset.Waist.corr(dataset.AT),'\n\n')

plt.plot(dataset.Waist,dataset["AT"],"ro");plt.xlabel("Waist");plt.ylabel("");plt.show()
