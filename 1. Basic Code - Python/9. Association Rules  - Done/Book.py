# import packages
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# print dataset
dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/9. Association Rules  - Done/book.csv")
dataset.columns
print(dataset)
new_dataset = dataset.columns

# print values
books = []
for i in dataset.columns:
    books.append(dataset[i].value_counts())
books_1 = []
for i in range(len(books)):
    books_1.append(list(books[i])[1])

# implement apriori algorithm & association rules
books_2 = apriori(dataset, min_support=0.005, max_len=3, use_colnames=True)
rules = association_rules(books_2, metric="lift", min_threshold=1)
rules.head(11)
rules.sort_values("lift", ascending=False, inplace=True)

#  convert string into unique integer values
books_2["itemsets"] = books_2["itemsets"].apply(lambda x: list(x)[0]).astype("unicode")
rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")

# plot bar graph
# new_dataset
plt.bar(new_dataset, height=books_1, color=['yellow', 'green'])
plt.xticks(new_dataset, rotation=90)
plt.xlabel("name")
plt.ylabel("count")

# itemsets
plt.bar(books_2.itemsets, height=books_2.support, color=['red'])
plt.xticks(books_2["itemsets"], rotation=90)
plt.xlabel('books')
plt.ylabel('support')
plt.show()

