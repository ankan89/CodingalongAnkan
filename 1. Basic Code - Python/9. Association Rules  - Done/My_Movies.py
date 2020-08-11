# import packages
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt
# print datasets
dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/9. Association Rules  - Done/my_movies.csv")
dataset.columns
print(dataset)
# split dataset
dataset1 = dataset.iloc[:,0:5]
dataset2 = dataset.iloc[:,5:]
dataset2.columns
print(dataset2)
movie = dataset2.columns
# function for print values
movie1 = []
for i in dataset2.columns:
    movie1.append(dataset2[i].value_counts())
movie2 = []
for i in range(len(movie1)):
    movie2.append(list(movie1[i])[1])
# implement association rules & apriori algorithm
movies = apriori(dataset2,min_support=0.005,max_len=3,use_colnames=True)
rules = association_rules(movies,metric="lift",min_threshold=1)
rules.head()
rules.sort_values("lift",ascending=False,inplace=True)
# convert string into integer values with the help of lambda
movies["itemsets"] = movies["itemsets"].apply(lambda x: list(x)[0]).astype("unicode")
rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")

# plot bar graph
# itemsets
plt.bar(movies.itemsets,height=movies.support,color=['yellow'])
plt.xticks(movies["itemsets"],rotation=90)
plt.xlabel("movies")
plt.ylabel("support")
# mo
plt.bar(movie,height=movie2,color=['red','black'])
plt.xticks(movie,rotation=90)
plt.xlabel("NAME")
plt.ylabel("COUNT")
plt.show()