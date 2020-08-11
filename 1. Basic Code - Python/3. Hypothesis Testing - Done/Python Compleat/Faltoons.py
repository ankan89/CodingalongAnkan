
#import pacakages.
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2

#print dataset.
Dataset=pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/3. Hypothesis Testing - Done/Faltoons.csv")
# convert string into unique int values and count.
Dataset.loc[Dataset['Weekdays']=='Male','Weekdays']=1
Dataset.loc[Dataset['Weekdays']=='Female','Weekdays']=0
f = Dataset['Weekdays'].value_counts()

Dataset.loc[Dataset['Weekend']=='Male','Weekend']=1
Dataset.loc[Dataset['Weekend']=='Female','Weekend']=0
f1 = Dataset['Weekend'].value_counts()

# store count data in dataframe.
d={'Weekend':[f[0],f[1]],'Weekdays':[f1[0],f1[1]]}
faltoon=pd.DataFrame.from_dict(d)

# Normality test 
# Anderson-Darling test
fa=stats.anderson(faltoon["Weekend"],dist='gumbel')
pValue = fa[1]
print("p-value is: "+str(pValue),'\n')

fa1=stats.anderson(faltoon["Weekdays"],dist='gumbel')
pValue = fa1[1]
print("p-value is: "+str(pValue),'\n\n\n')
#
# chi-square test
# contingency table
new_dataset = faltoon
print(new_dataset,'\n')
stat, p, dof, expected = chi2_contingency(new_dataset)
print('dof=%d' % dof,'\n')
print(expected,'\n\n\n')
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0)\n\n\n')
else:
	print('Independent (fail to reject H0)\n\n\n')
# probability=0.950, critical=3.841, stat=15.434
# Dependent (reject H0)
# 
# apha = 0.05        
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)\n\n\n')
else:
	print('Independent (fail to reject H0)\n\n\n')
# significance=0.050, p=0.000
# Dependent (reject H0)





















