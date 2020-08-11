# import packages
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
# print dataset
comp = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/1. A Basic Statistics Level 1 - Done/Computer_Data.csv")
print(comp.columns)
co=comp.columns
print(co, '\n\n')
# print skewness & kurtosis
print("Skewness : \n",comp.skew(),'\n\n\n')
print("Kurtosis : \n",comp.kurt(),'\n\n\n')
# print max & min, plot histogram, bar and categorise.
# column price
print("Max Price : ",comp["price"].max(),'\n\n\n')
print("Min Price : ",comp["price"].min(),'\n\n\n')
plt.hist(comp['price'],edgecolor='k')
plt.xlabel('Price')
plt.show()

# speed
print("Max Speed : ",comp["speed"].max(),'\n\n\n')
print("Min Speed : ",comp["speed"].min(),'\n\n\n')
plt.hist(comp['speed'],edgecolor='k')
plt.xlabel('Speed')
plt.show()

# hd
print("Max hd : ",comp["hd"].max(),'\n\n\n')
print("Min hd : ",comp["hd"].min(),'\n\n\n')
h1 = [i for i in comp.hd if (i>=80) & (i<250)]
h2 = [i for i in comp.hd if (i>=250) & (i<500)]
h3 = [i for i in comp.hd if (i>=500) & (i<1000)]
h4 = [i for i in comp.hd if (i>=1000) & (i<1500)]
h5 = [i for i in comp.hd if (i>=1500) & (i<2150)]
H6 = [len(h1),len(h2),len(h3),len(h4),len(h5)]
H7 = ["80-250","250-500","500-1000","1000-1500","1500-2100"]
for i, v in enumerate(H6):
    plt.text(i-.25,
              v,
              H6[i],
              fontsize=18,
              color="red")
plt.bar(H7,H6)
plt.xticks(H7)
plt.xlabel('HD')
plt.show()

# ram
print("Max ram : ",comp["ram"].max(),'\n\n\n')
print("Min ram : ",comp["ram"].min(),'\n\n\n')
r1 = [i for i in comp.ram if (i>=2) & (i<8)]
r2 = [i for i in comp.ram if (i>=8) & (i<16)]
r3 = [i for i in comp.ram if (i>=16) & (i<32)]
R4 = [len(r1),len(r2),len(r3)]
R5 = ["2-8","8-16","16-32"]
for i, v in enumerate(R4):
    plt.text(i-.25,
              v,
             R4[i],
              fontsize=18,
              color="red")
plt.bar(R5,R4)
plt.xticks(R5)
plt.xlabel('RAM')
plt.show()

# screen
print("Max screen : ",comp["screen"].max(),'\n\n\n')
print("Min screen : ",comp["screen"].min(),'\n\n\n')
n1 = [i for i in comp.screen if i==14]
n2 = [i for i in comp.screen if i==15]
n3 = [i for i in comp.screen if i==17]
N4 = [len(n1),len(n2),len(n3)]
N5 = ["scr-14","scr-15","scr-17"]
for i, v in enumerate(N4):
    plt.text(i-.25,
              v,
             N4[i],
              fontsize=18,
              color="red")
plt.bar(N5,N4)
plt.xticks(N5)
plt.xlabel('SCREEN')
plt.show()


# print skewness,kurtosis and plot histogram, distplot, boxplot
# price
print('Skewness Price : ',comp['price'].skew(),'\n\n\n')
print('Kurtosis Price : ',comp['price'].kurt(),'\n\n\n')
plt.hist(comp['price'],edgecolor='k')
plt.xlabel('Price')
plt.show()
sns.distplot(comp['price'],hist=False)
plt.xlabel('Price')
plt.show()
plt.boxplot(comp['price'])
plt.xlabel('Price')
plt.show()

# speed
print('Skewness speed : ',comp['speed'].skew(),'\n\n\n')
print('Kurtosis speed : ',comp['speed'].kurt(),'\n\n\n')
plt.hist(comp['speed'],edgecolor='k')
plt.xlabel('Speed')
plt.show()
sns.distplot(comp['speed'],hist=False)
plt.xlabel('Speed')
plt.show()
plt.boxplot(comp['speed'])
plt.xlabel('Speed')
plt.show()

# hd
print('Skewness hd : ',comp['hd'].skew(),'\n\n\n')
print('Kurtosis hd : ',comp['hd'].kurt(),'\n\n\n')
plt.hist(comp['hd'],edgecolor='k')
plt.xlabel('HD')
plt.show()
sns.distplot(comp['hd'],hist=False)
plt.xlabel('HD')
plt.show()
plt.boxplot(comp['hd'])
plt.xlabel('HD')
plt.show()

# ram
print('Skewness ram : ',comp['ram'].skew(),'\n\n\n')
print('Kurtosis ram : ',comp['ram'].kurt(),'\n\n\n')
plt.hist(comp['ram'],edgecolor='k')
plt.xlabel('RAM')
plt.show()
sns.distplot(comp['ram'],hist=False)
plt.xlabel('RAM')
plt.show()
plt.boxplot(comp['ram'])
plt.xlabel('RAM')
plt.show()

# screen
print('Skewness screen : ',comp['screen'].skew(),'\n\n\n')
print('Kurtosis screen : ',comp['screen'].kurt(),'\n\n\n')
plt.hist(comp['screen'],edgecolor='k')
plt.xlabel('SCREEN')
plt.show()
sns.distplot(comp['screen'],hist=False)
plt.xlabel('SCREEN')
plt.show()
plt.boxplot(comp['screen'])
plt.xlabel('SCREEN')
plt.show()

# plot graph between price & screen
df=pd.crosstab(comp['price'],comp['screen'])
df.plot(kind="line",stacked=True)
plt.grid(axis="y")
plt.show()

# speed & screen
df=pd.crosstab(comp['speed'],comp['screen'])
df.plot(kind="bar",stacked=True)
plt.grid(axis="y")
plt.show()

# hd & screen
df=pd.crosstab(comp['hd'],comp['screen'])
df.plot(kind="line",stacked=True)
plt.grid(axis="y")
plt.show()

# ram & screen
df=pd.crosstab(comp['ram'],comp['screen'])
df.plot(kind="bar",stacked=True)
plt.grid(axis="y")
plt.show()











































































