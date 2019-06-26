import numpy as np
import pandas

#result = isolate a target series
target = d.loc[:,'parfam']

#features = quasi sentence composition in manifesto
#isolate df with only policy quasi-sentences
quasi = d.filter(like='per')
quasi = quasi.iloc[:,2:-1]

### UNSUPERVISED ML
### Algorithm 1: K-means clustering
from sklearn import cluster, preprocessing
testdf = pd.concat([target[1:],quasi])
testdf = testdf.fillna(value=999) #replace all NAs with 999

#preprocessing: turn labels into numeric values s.t. k_means takes them
le = preprocessing.LabelEncoder()
for column_name in testdf.columns:
        if testdf[column_name].dtype == object:
            testdf[column_name] = le.fit_transform(testdf[column_name].astype(str))
        else:
            pass

k_means = cluster.KMeans(n_clusters=10) #follow the codebook. assume some 10 groups
k_means.fit(testdf)

#check the output groups
clustered = k_means.labels_
print(clustered)
print(len(clustered)) #why are we getting different N?

#extract output
y = k_means.predict(testdf)

#plot the clusters : 2D example
#try a 3D plot: plot on the following: per401 (free market economy), per504 (welfare state expansion), per601 (national way of life positive)
import matplotlib.pyplot as plt

plt.scatter(testdf.loc[:, 'per401'], testdf.loc[:, 'per504'], c=y, s=5, cmap='viridis', alpha=0.1)

centers = k_means.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.xlabel('Free Market Economy'); plt.ylabel('Welfare State Expansion'); plt.title('K-means clustering of political parties. K=10')
plt.grid(True) #to turn on the grid
plt.show()
#plt.scatter(testdf.loc[:,"per401"], testdf[:, 1], s=50);

#plot the clusters: 3D example
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(testdf.loc[:, 'per401'], testdf.loc[:, 'per504'], testdf.loc[:,'per601'], c=y, cmap="hsv", marker='o', alpha=0.05)
centers = k_means.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], centers[:,2], c='black', s=200, alpha=0.5);
ax.set_xlabel('Free Market Economy')
ax.set_ylabel('Welfare State Expansion')
ax.set_zlabel('National Way of Life: Positive')
plt.title('K-means clustering of political parties. K=10')

plt.show()

#next steps: cut down K/ try other algorithms
