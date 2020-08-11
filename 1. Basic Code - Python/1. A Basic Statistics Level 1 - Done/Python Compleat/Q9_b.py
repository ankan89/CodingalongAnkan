# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# print dataset
dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/1. A Basic Statistics Level 1 - Done/Q9_b.csv")
print(dataset.columns,'\n\n')
new_dataset=dataset.columns
print(new_dataset,'\n\n')

# print skewnesws & kurtosis
print('Skewness : \n',dataset.skew(),'\n\n')
print('Kurtosis : \n',dataset.kurt(),'\n\n')
print('Correlation : \n',dataset.corr(),'\n\n')
print('Dataset Head : \n',dataset.head(),'\n\n')

# print max & min, plot histogram, bar and categorise.
# column SP
print("Max SP : ",dataset["SP"].max(),'\n\n\n')
print("Min SP : ",dataset["SP"].min(),'\n\n\n')
plt.hist(dataset['SP'],edgecolor='k')
plt.xlabel('SP')
plt.show()

# column WT
print("Max WT : ",dataset["WT"].max(),'\n\n\n')
print("Min WT : ",dataset["WT"].min(),'\n\n\n')
plt.hist(dataset['WT'],edgecolor='k')
plt.xlabel('WT')
plt.show()


# print skewness,kurtosis and plot histogram, distplot, boxplot
# SP
print('Skewness SP : ',dataset['SP'].skew(),'\n\n\n')
print('Kurtosis SP : ',dataset['SP'].kurt(),'\n\n\n')
plt.hist(dataset['SP'],edgecolor='k')
plt.xlabel('SP')
plt.show()
sns.distplot(dataset['SP'],hist=False)
plt.xlabel('SP')
plt.show()
plt.boxplot(dataset['SP'])
plt.xlabel('SP')
plt.show()

# WT
print('Skewness WT : ',dataset['WT'].skew(),'\n\n\n')
print('Kurtosis WT : ',dataset['WT'].kurt(),'\n\n\n')
plt.hist(dataset['WT'],edgecolor='k')
plt.xlabel('WT')
plt.show()
sns.distplot(dataset['WT'],hist=False)
plt.xlabel('WT')
plt.show()
plt.boxplot(dataset['WT'])
plt.xlabel('WT')
plt.show()

# plot graph between SP and WT
df=pd.crosstab(dataset['SP'],dataset['WT'])
df.plot(kind="line",stacked=True,)
plt.grid(axis="y")
plt.show()

# visulization
plt.plot(np.arange(81),dataset.SP)
plt.plot(np.arange(81),dataset.WT)
plt.show()
plt.plot(np.arange(81),dataset.SP,"ro-")
plt.plot(np.arange(81),dataset.WT,"ro-")
plt.show()
dataset.SP.value_counts().plot(kind="pie")
plt.show()
dataset.SP.value_counts().plot(kind="bar")
plt.show()
print('Corr SP',dataset.SP.corr(dataset.SP),'\n\n')
print('Corr WT',dataset.SP.corr(dataset.WT),'\n\n')

plt.plot(dataset.SP,dataset["WT"],"ro");plt.xlabel("SP");plt.ylabel("WT");plt.show()
