
#import pacakages.
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2


#print dataset.
dataset=pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/3. Hypothesis Testing/Costomer+OrderForm.csv")
# convert string into unique int values and count.
dataset.loc[dataset['Phillippines']=='Error Free','Phillippines']=1
dataset.loc[dataset['Phillippines']=='Defective','Phillippines']=0
c = dataset['Phillippines'].value_counts()

dataset.loc[dataset['Indonesia']=='Error Free','Indonesia']=1
dataset.loc[dataset['Indonesia']=='Defective','Indonesia']=0
c1 = dataset['Indonesia'].value_counts()

dataset.loc[dataset['Malta']=='Error Free','Malta']=1
dataset.loc[dataset['Malta']=='Defective','Malta']=0
c2 = dataset['Malta'].value_counts()

dataset.loc[dataset['India']=='Error Free','India']=1
dataset.loc[dataset['India']=='Defective','India']=0
c3 = dataset['India'].value_counts()

# store count data in dataframe.
new_dataset={'Phillippines':[c[0],c[1]],'Indonesia':[c1[0],c1[1]],'Malta':[c2[0],c2[1]],'India':[c3[0],c3[1]]}
customerorder=pd.DataFrame.from_dict(new_dataset)

# Normality test 
# Anderson-Darling test
cu=stats.anderson(customerorder["Phillippines"],dist='gumbel')
pValue = cu[1]
print(
	"p-value is: "+str(pValue),'\n')


cu1=stats.anderson(customerorder["Indonesia"],dist='gumbel')
pValue = cu1[1]
print(
	"p-value is: "+str(pValue),'\n')


cu3=stats.anderson(customerorder["India"],dist='gumbel')
pValue = cu3[1]
print(
	"p-value is: "+str(pValue),'\n\n\n')


#
# chi-square test
# contingency table
customerorder1 = customerorder
print(customerorder1,'\n\n\n')
stat, p, dof, expected = chi2_contingency(customerorder1)
print('dof=%d' % dof,'\n')
print(expected,'\n\n\n')
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0)','\n')
else:
	print('Independent (fail to reject H0)','\n')
# probability=0.950, critical=7.815, stat=3.859
# Independent (fail to reject H0)
# 
# apha = 0.05        
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)','\n')
else:
	print('Independent (fail to reject H0)','\n')
# significance=0.050, p=0.277
# Independent (fail to reject H0)













