# import packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# prin dataset
dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/11. Recommendation system - Done/book1.csv",encoding = "ISO-8859-1")
print(dataset.shape,'\n\n\n')
print(dataset.columns,'\n\n\n')
dataset.drop('Unnamed: 0',axis=1,inplace=True)
# function for dataset rating
def rate(m):
    h=dataset['Book.Rating']
    reco=dataset[dataset['Book.Rating']==m]
    return reco
v=rate(5)
# function for dataset title
def boo(name):
    k=dataset['Title']
    l=dataset['Book.Rating']
    ind=0
    for i in range(len(k)):
        if(name==k[i]):
            ind=i
    r=l[ind]
    reco=dataset[dataset['Book.Rating']==r]
    return reco
# print according to rating
z='More Cunning Than Man: A Social History of Rats and Man'
final =boo(z)
print(final)




