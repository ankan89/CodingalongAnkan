import pandas as pd
import scipy
from scipy import stats
import seaborn as sns

# print datasets
dataset=pd.read_csv("/Users/Anabcan/PycharmProjects/Basiccode/1. Assignments/3. Hypothesis Testing - Done/Cutlets.csv")
dataset.columns = "unitA","unitB"

# Normality test 
#Shapiro Test
# unit A
dataset1=stats.shapiro(dataset["unitA"])
sns.distplot(dataset["unitA"])
dataset1_pValue = dataset1[1]
print("p-value is: "+str(dataset1_pValue),'\n')
# unit B
dataset2=stats.shapiro(dataset["unitB"])
dataset2_pValue=dataset2[1]
sns.distplot(dataset["unitB"])
print("p-value is: "+str(dataset2_pValue),'\n\n\n')

#Varience Test
cut1 = scipy.stats.levene(dataset["unitA"], dataset["unitB"])
cut1

cut2 = scipy.stats.ttest_ind(dataset["unitA"], dataset["unitB"])
cut2

c1 ={"dataset1":[dataset1[0],dataset1[1]],"dataset2":[dataset2[0],dataset2[1]],"cut1":[cut1[0],cut1[1]],"cut2":[cut2[0],cut2[1]]}
c1
dataset_ab = pd.DataFrame.from_dict(c1)
dataset_ab

#hypothesis testing
#ANOVA test
cutresult = stats.f_oneway(dataset["unitA"], dataset["unitB"])
print(cutresult[1],'\n\n\n')
# function for print Ho / Ha
if cutresult[1] < 0.05:
  abc='(Reject H0) - Not all sizes are equal'
else:
 abc='(fail to reject H0) - All sizes are equal'
print(abc)






















































