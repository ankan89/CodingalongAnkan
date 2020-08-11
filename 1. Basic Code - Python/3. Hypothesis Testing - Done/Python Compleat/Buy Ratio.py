# import pacakages.
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2

#print dataset.
dataset=pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/3. Hypothesis Testing - Done/BuyerRatio.csv")

# Normality test 
# Anderson-Darling test
east=stats.anderson(dataset["East"],dist='gumbel')
pValue_east = east[1]
print("p-value is: "+str(pValue_east),'\n')

east=stats.anderson(dataset["West"],dist='gumbel')
pValue_east = east[1]
print("p-value is: "+str(pValue_east),'\n')

east=stats.anderson(dataset["North"],dist='gumbel')
pValue_east = east[1]
print("p-value is: "+str(pValue_east),'\n')

east=stats.anderson(dataset["South"],dist='gumbel')
pValue_east = east[1]
print("p-value is: "+str(pValue_east),'\n\n')

#hypothesis testing

#ANOVA test
result_east = stats.f_oneway(dataset["East"],dataset["West"],dataset["North"],dataset["South"])
print('ANOVA test : ',result_east[1],'\n\n')
#pvalue < 0.05 --- reject null hypothesis.

# chi-square test
# contingency table
buyratio = dataset.iloc[:,1:]
print(buyratio,'\n')
stat, p, dof, expected = chi2_contingency(buyratio)
print('dof=%d' % dof,'\n')
print(expected,'\n\n')

# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# probability=0.950, critical=7.815, stat=1.596
# Independent (fail to reject H0)
# 
# apha = 0.05
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# significance=0.05, p=0.660,  p-Value > alpha
# Independent (fail to reject H0),  All proportions are equal









