import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import	AgglomerativeClustering


dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/8. PCA - Done/wine.csv")



############################ ----Data Standardization---- ##############################

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dataset)
scaler_dataset = scaler.transform(dataset)

# #############_______________________________________________________________________________________________________________###############
# #############_______________________________________________________________________________________________________________###############
# ############# Type 5 --> Principle Component Analysis (PCA) ##########
# #############_______________________________________________________________________________________________________________###############
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(scaler_dataset)
pca_dataset = pca.transform(scaler_dataset)

print(scaler_dataset.shape,'\n\n\n')

print(pca_dataset.shape,'\n\n\n')

print(dataset.describe(),'\n\n\n')

print(dataset.head(),'\n\n\n')

# # ########################### Model Analysis ##########################
# #   ######## As it is a feature optimisation process so we will try to understand the best fit ##########

plt.figure(figsize=(8,6))
plt.scatter(pca_dataset[:,0],pca_dataset[:,1],c=dataset['Type'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.show()

# ############# Determining the best PCA value and plotting it #############

print(pca.components_,"\n\n\n")
components_dataset = pd.DataFrame(pca.components_,columns=dataset.columns)
print(components_dataset,"\n\n\n")

plt.figure(figsize=(14,8))
sns.heatmap(components_dataset,cmap='plasma')
plt.show()

# Considering only numerical data
dataset.data = dataset.iloc[:,1:]
print(dataset.data.head(4))

# Normalizing the numerical data
wine_normal_dataset = scale(dataset.data)
pca = PCA(n_components = 6)
pca_values = pca.fit_transform(wine_normal_dataset)

# The amount of variance that each PCA explains
var = pca.explained_variance_ratio_
print(var)
pca.components_[0]

# Cumulative variance
var1 = np.cumsum(np.round(var,decimals = 4)*100)
print(var1)

# Variance plot for PCA components obtained
plt.plot(var1,color="red")
plt.xlabel('Variance')
plt.show()

# scatter plot between PCA1 and PCA2
x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:,2]
plt.scatter(x,y,color=["blue"])
plt.xlabel("PCA")
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(np.array(x),np.array(y),np.array(z),c=["green"])
plt.xlabel("PCA 3D")
plt.show()

# Normalization function
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
norm_dataset = norm_func(dataset.iloc[:,1:])
print(norm_dataset.head(10))  # Top 10 rows

# K means clustering
# plot elbow curve
k = list(range(2,15))

TWSS = [] # variable for storing total within sum of squares for each kmeans
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(norm_dataset)
    WSS = [] # variable for storing within sum of squares for each cluster
    for j in range(i):
        WSS.append(sum(cdist(norm_dataset.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,norm_dataset.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k);plt.show()

# Selecting clusters from the above scree plot which is the optimum number of clusters
model=KMeans(n_clusters=4)
model.fit(norm_dataset)
model.labels_ # getting the labels of clusters assigned to each row
md=pd.Series(model.labels_)  # converting numpy array into pandas series object
dataset['clust']=md # creating a  new column and assigning it to new column
print(norm_dataset.head())

# hierarchical clustering
print(type(norm_dataset))
z = linkage(norm_dataset, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# apply AgglomerativeClustering
h_complete	=	AgglomerativeClustering(n_clusters=4,	linkage='complete',affinity = "euclidean").fit(norm_dataset)
cluster_labels=pd.Series(h_complete.labels_)
dataset['clust']=cluster_labels # creating a  new column and assigning it to new column
print(dataset.head())

