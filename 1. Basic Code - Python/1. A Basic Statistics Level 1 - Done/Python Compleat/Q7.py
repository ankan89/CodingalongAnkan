# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# print dataset
cars = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/1. A Basic Statistics Level 1 - Done/Q7.csv")
print(cars.columns,'\n\n')
cars = cars.rename(columns = {'Unnamed: 0': 'Model'})
dataset=cars.columns
print(dataset,'\n\n')

# print skewnesws & kurtosis
print('Skewness : \n',cars.skew(),'\n\n')
print('Kurtosis : \n',cars.kurt(),'\n\n')
print('Correlation : \n',cars.corr(),'\n\n')
print('Dataset Head : \n',cars.head(),'\n\n')

# print max & min, plot histogram, bar and categorise.
# column points
print("Max Points : ",cars["Points"].max(),'\n\n\n')
print("Min Points : ",cars["Points"].min(),'\n\n\n')
plt.hist(cars['Points'],edgecolor='k')
p1 = [i for i in cars.Points if (i>=2) & (i<3)]
p2 = [i for i in cars.Points if (i>=3) & (i<4)]
p3 = [i for i in cars.Points if (i>=4) & (i<5)]
P4 = [len(p1),len(p2),len(p3)]
P5 = ["2-3","3-4","4-5"]
for i, v in enumerate(P4):
    plt.text(i-.25,
              v,
              P4[i],
              fontsize=18,
              color="red")
plt.bar(P5,P4)
plt.xticks(P5,rotation=90)
plt.xlabel('Points')
plt.show()

# score
print("Max Score : ",cars["Score"].max(),'\n\n\n')
print("Min Score : ",cars["Score"].min(),'\n\n\n')
plt.hist(cars['Score'],edgecolor='k')
s1 = [i for i in cars.Score if (i>=1) & (i<2.5)]
s2 = [i for i in cars.Score if (i>=2.5) & (i<3.5)]
s3 = [i for i in cars.Score if (i>=3.5) & (i<5.5)]
S4 = [len(s1),len(s2),len(s3)]
S5 = ["2-3","3-4","4-5"]
for i, v in enumerate(S4):
    plt.text(i-.25,
              v,
              S4[i],
              fontsize=18,
              color="red")
plt.bar(S5,S4)
plt.xticks(S5,rotation=90)
plt.xlabel('Score')
plt.show()

# weigh
print("Max Weigh : ",cars["Weigh"].max(),'\n\n\n')
print("Min Weigh : ",cars["Weigh"].min(),'\n\n\n')
plt.hist(cars['Weigh'],edgecolor='k')
w1 = [i for i in cars.Weigh if (i>=14) & (i<17)]
w2 = [i for i in cars.Weigh if (i>=17) & (i<19)]
w3 = [i for i in cars.Weigh if (i>=19) & (i<25)]
W4 = [len(w1),len(w2),len(w3)]
W5 = ["14-17","17-19","19-25"]
for i, v in enumerate(W4):
    plt.text(i-.25,
              v,
              W4[i],
              fontsize=18,
              color="red")
plt.bar(W5,W4)
plt.xticks(W5,rotation=90)
plt.xlabel('Weigh')
plt.show()

# print skewness,kurtosis and plot histogram, distplot, boxplot
# points
print('Skewness Points : ',cars['Points'].skew(),'\n\n\n')
print('Kurtosis Points : ',cars['Points'].kurt(),'\n\n\n')
plt.hist(cars['Points'],edgecolor='k')
plt.xlabel('Points')
plt.show()
sns.distplot(cars['Points'],hist=False)
plt.xlabel('Points')
plt.show()
plt.boxplot(cars['Points'])
plt.xlabel('Points')
plt.show()

# score
print('Skewness Score : ',cars['Score'].skew(),'\n\n\n')
print('Kurtosis Score : ',cars['Score'].kurt(),'\n\n\n')
plt.hist(cars['Score'],edgecolor='k')
plt.xlabel('Score')
plt.show()
sns.distplot(cars['Score'],hist=False)
plt.xlabel('Score')
plt.show()
plt.boxplot(cars['Score'])
plt.xlabel('Score')
plt.show()

# weigh
print('Skewness Weigh : ',cars['Weigh'].skew(),'\n\n\n')
print('Kurtosis Weigh : ',cars['Weigh'].kurt(),'\n\n\n')
plt.hist(cars['Weigh'],edgecolor='k')
plt.xlabel('Weigh')
plt.show()
sns.distplot(cars['Weigh'],hist=False)
plt.xlabel('Weigh')
plt.show()
plt.boxplot(cars['Weigh'])
plt.xlabel('Weigh')
plt.show()

# plot graph between point and model
df=pd.crosstab(cars['Points'],cars['Model'])
df.plot(kind="bar",stacked=True,)
plt.grid(axis="y")
plt.show()

# visulization
plt.plot(np.arange(32),cars.Points)
plt.plot(np.arange(32),cars.Score)
plt.plot(np.arange(32),cars.Weigh)
plt.show()
plt.plot(np.arange(32),cars.Points,"ro-")
plt.plot(np.arange(32),cars.Score,"ro-")
plt.plot(np.arange(32),cars.Weigh,"ro-")
plt.show()
cars.Points.value_counts().plot(kind="pie")
plt.show()
cars.Points.value_counts().plot(kind="bar")
plt.show()
print('Corr Point',cars.Points.corr(cars.Points),'\n\n')
print('Corr Score',cars.Score.corr(cars.Score),'\n\n')
print('Corr Weigh',cars.Points.corr(cars.Weigh),'\n\n')

plt.plot(cars.Points,cars["Score"],"ro");plt.xlabel("Points");plt.ylabel("Score");plt.show()









































