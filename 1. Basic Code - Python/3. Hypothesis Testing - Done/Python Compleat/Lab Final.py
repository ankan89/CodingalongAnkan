
#import pacakages.
import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import seaborn as sns

#print dataset.
labtat=pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/3. Hypothesis Testing - Done/LabTAT.csv")

#Normality test 
#Shapiro Test
labtat1=stats.shapiro(labtat["Laboratory 1"])
sns.distplot(labtat["Laboratory 1"])
labtat1_pValue = labtat1[1]
print("p-value is: "+str(labtat1_pValue),'\n')

labtat2=stats.shapiro(labtat["Laboratory 2"])
sns.distplot(labtat["Laboratory 2"])
labtat2_pValue = labtat2[1]
print("p-value is: "+str(labtat2_pValue),'\n')

labtat3=stats.shapiro(labtat["Laboratory 3"])
sns.distplot(labtat["Laboratory 3"])
labtat3_pValue = labtat3[1]
print("p-value is: "+str(labtat3_pValue),'\n')

labtat4=stats.shapiro(labtat["Laboratory 4"])
sns.distplot(labtat["Laboratory 4"])
labtat4_pValue = labtat4[1]
print("p-value is: "+str(labtat4_pValue),'\n\n\n')

#Varience Test
scipy.stats.levene(labtat["Laboratory 1"], labtat["Laboratory 2"], labtat["Laboratory 3"], labtat["Laboratory 4"])

#hypothesis testing
#ANOVA test
labresult = stats.f_oneway(labtat["Laboratory 1"], labtat["Laboratory 2"], labtat["Laboratory 3"], labtat["Laboratory 4"])
print(labresult[1],'\n\n')
 
#pvalue < 0.05 --- reject null hypothesis.

# function for print Ho / Ha
if labresult[1] < 0.05:
  abc='(Reject H0) - There is no difference in TAT'
else:
 abc='(fail to reject H0) - There is difference in TAT'
print(abc)



















