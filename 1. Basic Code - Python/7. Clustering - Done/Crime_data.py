import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import	AgglomerativeClustering

dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/7. Clustering - Done/crime_data.csv")


######## Basic Analysis ################

print(dataset.head(),'\n\n\n')

print(dataset.info(),'\n\n\n')

print(dataset.describe(),'\n\n\n')

print(dataset.columns,'\n\n\n')

print(dataset.shape,'\n\n\n')

############################ ----Data Normalisation---- ##############################

def norm_func(i):
    x =(i*i.min())/ (i.max() - i.min())
    return(x)

norm_dataset = norm_func(dataset.iloc[:,1:])
print(norm_dataset.head(),'\n\n\n')


#############_______________________________________________________________________________________________________________###############
#############_______________________________________________________________________________________________________________###############
############# Type 3 --> Clustering (Kmeans) ##########
#############_______________________________________________________________________________________________________________###############
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
Model_1= kmeans.fit(dataset.drop('Unnamed: 0',axis=1))

# ########################### Model Analysis ##########################
#   ######## As it is a classification process so we will try to understand the best fit ##########

print(kmeans.cluster_centers_,'\n\n\n')
print(kmeans.labels_,'\n\n\n')


############# Determining the best K value and plotting it #############

############# Hierarchical clustering #####################

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage

z = linkage(dataset.iloc[:,1:], method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,leaf_rotation=0.,leaf_font_size=8.,)
plt.show()

############### k means #####################

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

k = list(range(2, 15))

WSS1 = []
TWSS = []

for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(dataset.drop('Unnamed: 0', axis=1))
    WSS = []
    for j in range(i):
        WSS.append(sum(cdist(norm_dataset.iloc[kmeans.labels_ == j, :],
                             kmeans.cluster_centers_[j]
                             .reshape(1, norm_dataset.shape[1]), "euclidean")))
    TWSS.append(sum(WSS))
    WSS1.append(np.mean(WSS))

fig, (axes1,axes2) = plt.subplots(1, 2,figsize=(10,8))
axes1.plot(k, WSS1, 'ro-')
axes1.set_xlabel('No of Cluster')
axes1.set_ylabel('Total Mean WSS')
axes1.set_title('K')
axes2.plot(k,TWSS,'bo-')
axes2.set_xlabel('No of Cluster')
axes2.set_ylabel('Total WSS')
axes2.set_title('K')
plt.show()



############# Re-creating the model with the best K value #############

model = KMeans(n_clusters=3)
model.fit(norm_dataset)
nd = pd.Series(model.labels_)
dataset['Clust'] = nd
print(dataset.head(),'\n\n\n')

groupby = pd.DataFrame(dataset.iloc[:,:].groupby(dataset.Clust).mean())
print(groupby,'\n\n\n')

a = 3  ####### Minimum as per the groupby output ( Variable a is considered as 3 based on the cluster nos corresponding to the least value in the minimum feature) ########

print('Few of the best places to reside \n\n',dataset[dataset['Clust']==a]['Unnamed: 0'])



#############_______________________________________________________________________________________________________________#################################################################################

# k means
k = list(range(2,8))

TWSS = []
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(norm_dataset)
    WSS = []
    for j in range(i):
        WSS.append(sum(cdist(norm_dataset.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,norm_dataset.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
#  Scree plot
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k);plt.show()

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters
model=KMeans(n_clusters=5)
model.fit(norm_dataset)
model.labels_
md=pd.Series(model.labels_)
dataset['clust']=md
print(dataset.head())
print(norm_dataset.head())
dataset = dataset.iloc[:,[6,0,1,2,3,4,5]]
dataset.iloc[:,1:7].groupby(dataset.clust).mean()
dataset.to_csv("crime.csv")

# Hierarchical clustering
z = linkage(norm_dataset, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,
    leaf_font_size=8.,
)
plt.show()

# AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(norm_dataset)
cluster_labels= pd.Series(h_complete.labels_)
dataset['clust']=cluster_labels
print(dataset.head())