# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# print dataset
car = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/1. A Basic Statistics Level 1 - Done/Cars.csv")
car.columns
ca=car.columns
print(ca,'\n\n')
# print skewness, kurtosis, correlation
print("Skewness \n",car.skew(),'\n\n\n')
print("Kurtosis \n",car.kurt(),"\n\n\n")
print("Correalation \n",car.corr(),'\n\n\n')
print(car.head(),'\n\n\n')
# print max & min, plot histogram, bar and categorise.
# column HP
print("Max HP : ",car["HP"].max(),'\n\n\n')
print("Min HP : ",car["HP"].min(),'\n\n\n')
plt.hist(car['HP'],edgecolor='k')
plt.xlabel('HP')
plt.show()

# MPG
print("Max MPG : ",car["MPG"].max(),'\n\n\n')
print("Max MPG : ",car["MPG"].min(),'\n\n\n')
plt.hist(car['MPG'],edgecolor='k')
plt.xlabel('MPG')
plt.show()

# VOL
print("Max VOL : ",car["VOL"].max(),'\n\n\n')
print("Min VOL : ",car["VOL"].min(),'\n\n\n')
plt.hist(car['VOL'],edgecolor='k')
plt.xlabel('VOL')
plt.show()

# SP
print("Max SP : ",car["SP"].max(),'\n\n\n')
print("Max SP : ",car["SP"].min(),'\n\n\n')
plt.hist(car['SP'],edgecolor='k')
plt.xlabel('SP')
plt.show()

# WT
print("Max WT : ",car["WT"].max(),'\n\n\n')
print("Max WT : ",car["WT"].min(),'\n\n\n')
plt.hist(car['WT'],edgecolor='k')
plt.xlabel('WT')
plt.show()

# print skewness,kurtosis and plot histogram, distplot, boxplot
# HP
print("Skewness HP : ",car['HP'].skew(),'\n\n\n')
print("Kurtosis HP : ",car['HP'].kurt(),'\n\n\n')
plt.hist(car['HP'],edgecolor='k')
plt.xlabel('HP')
plt.show()
sns.distplot(car['HP'],hist=False)
plt.xlabel('HP')
plt.show()
plt.boxplot(car['HP'])
plt.xlabel('HP')
plt.show()
# MPG
print("Skewness MPG : ",car['MPG'].skew(),'\n\n\n')
print("Kurtosis MPG : ",car['MPG'].kurt(),'\n\n\n')
plt.hist(car['MPG'],edgecolor='k')
plt.xlabel('MPG')
plt.show()
sns.distplot(car['MPG'],hist=False)
plt.xlabel('MPG')
plt.show()
plt.boxplot(car['MPG'])
plt.xlabel('MPG')
plt.show()
# VOL
print("Skewness VOL : ",car['VOL'].skew(),'\n\n\n')
print("Kurtosis VOL : ",car['VOL'].kurt(),'\n\n\n')
plt.hist(car['VOL'],edgecolor='k')
plt.xlabel('VOL')
plt.show()
sns.distplot(car['VOL'],hist=False)
plt.xlabel('VOL')
plt.show()
plt.boxplot(car['VOL'])
plt.xlabel('VOL')
plt.show()
# SP
print("Skewness SP : ",car['SP'].skew(),'\n\n\n')
print("Kurtosis SP : ",car['SP'].kurt(),'\n\n\n')
plt.hist(car['SP'],edgecolor='k')
plt.xlabel('SP')
plt.show()
sns.distplot(car['SP'],hist=False)
plt.xlabel('SP')
plt.show()
plt.boxplot(car['SP'])
plt.xlabel('SP')
plt.show()
# WT
print("Skewness WT : ",car['WT'].skew(),'\n\n\n')
print("Kurtosis WT : ",car['WT'].kurt(),'\n\n\n')
plt.hist(car['WT'],edgecolor='k')
plt.xlabel('WT')
plt.show()
sns.distplot(car['WT'],hist=False)
plt.xlabel('WT')
plt.show()
plt.boxplot(car['WT'])
plt.xlabel('WT')
plt.show()

# visulization
plt.plot(np.arange(81),car.HP)
plt.plot(np.arange(81),car.MPG)
plt.plot(np.arange(81),car.VOL)
plt.plot(np.arange(81),car.SP)
plt.plot(np.arange(81),car.WT)
plt.show()
print("Correlation HP~HP",car.HP.corr(car.HP),'\n')
print("Correlation HP~MPG",car.HP.corr(car.MPG),'\n')
print("Correlation HP~VOL",car.HP.corr(car.VOL),'\n')
print("Correlation HP~SP",car.HP.corr(car.SP),'\n')
print("Correlation HP~WT",car.HP.corr(car.WT),'\n')
plt.plot(np.arange(81),car.HP,"ro-")
plt.xlabel('HP')
plt.show()
plt.plot(np.arange(81),car.MPG,"ro-")
plt.xlabel('MPG')
plt.show()
plt.plot(np.arange(81),car.VOL,"ro-")
plt.xlabel('VOL')
plt.show()
plt.plot(np.arange(81),car.SP,"ro-")
plt.xlabel('SP')
plt.show()
plt.plot(np.arange(81),car.WT,"ro-")
plt.xlabel('WT')
plt.show()
car.VOL.value_counts().plot(kind="pie")
plt.xlabel('By VOL')
plt.show()
car.VOL.value_counts().plot(kind="bar")
plt.xlabel('By VOL')
plt.show()
sns.pairplot(car,hue="VOL",size=3)
plt.show()
sns.pairplot(car.iloc[:,0:80])
plt.show()
plt.plot(car.HP,car["SP"],"ro");plt.xlabel("HP");plt.ylabel("SP")
plt.show()





























