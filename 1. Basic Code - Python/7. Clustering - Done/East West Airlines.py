# import pacakegs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import	AgglomerativeClustering

# print dataset
dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/7. Clustering - Done/EastWestAirlines.csv")
dataset.columns
new_dataset=dataset.columns
print(new_dataset,'\n\n\n')

# apply normalization function on dataset.
def norm_func(i):
    x = (i-i.min())	/ (i.max() - i.min())
    return (x)
norm_dataset = norm_func(dataset.iloc[:,1:])
print(norm_dataset.head(),'\n\n\n')

# k means
k = list(range(2,15))

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
print(model.labels_)
md=pd.Series(model.labels_)
dataset['clust']=md
print(norm_dataset.head())
dataset = dataset.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
dataset.iloc[:,1:7].groupby(dataset.clust).mean()
dataset.to_csv("eastair.csv")


# Hierarchical clustering
# apply normalization function on dataset.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
norm_dataset = norm_func(dataset.iloc[:,1:])
print(type(norm_dataset))

# Hierarchical clustering.
# p = np.array(norm_dataset) # converting into numpy array format
# plot dendogram  # linkage method = complete, distance method = euclidean
z = linkage(norm_dataset, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,
    leaf_font_size=8.,
)
plt.show()

# plot dendogram  # linkage method = average, distance method = cosine
z1 = linkage(norm_dataset, method="average",metric="cosine")
plt.figure(figsize=(20,5));plt.title('Hierarchical Clustering Dendrogram 1');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z1,
    leaf_rotation=0.,
    leaf_font_size=8.,
)
plt.show()

# plot dendogram  # linkage method = median, distance method = euclidean
z2 = linkage(norm_dataset, method="median",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram 2');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z2,
    leaf_rotation=0.,
    leaf_font_size=8.,
)
plt.show()

# AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(norm_dataset)
cluster_labels= pd.Series(h_complete.labels_)
dataset['clust']=cluster_labels
print(dataset.head())