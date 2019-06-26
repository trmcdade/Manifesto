#PCA alt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from IPython.core.display import display
from sklearn.decomposition import PCA
#seperate data from outcome!
y = np.array(d.iloc[1:,9])
pca_data = d.iloc[1:,10:len(d.columns)]
for column_name in pca_data.columns:
       if pca_data[column_name].dtype == object:
           pca_data[column_name] = le.fit_transform(pca_data[column_name].astype(str))
       else:
           pass

pca_data.head()

## do PCA
ncomponents = 40
pca = PCA(n_components = ncomponents)
pca.fit(pca_data)
X_pca = pca.transform(pca_data)
print("original shape:   ", pca_data.shape)
print("transformed shape:", X_pca.shape)

## reduced dataset
reduced = np.column_stack((y,X_pca))

##Re-fit K-meansk_means = cluster.KMeans(n_clusters=10) #follow the codebook. assume some 10 groups
k_means = cluster.KMeans(n_clusters=len(np.unique(y)))
k_means.fit(reduced)

#check the output groups
clustered = k_means.labels_
print(clustered)

#extract output
y_clusters = k_means.predict(reduced)

#plot the clusters : 2D example
#try a 3D plot: plot on the following: per401 (free market economy), per504 (welfare state expansion), per601 (national way of life positive)
import matplotlib
import matplotlib.pyplot as plt

#plot the clusters: 3D example
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced[:,1], reduced[:,2], reduced[:,3], c=y_clusters, cmap="hsv", marker='o', alpha=0.05)

centers = k_means.cluster_centers_
ax.scatter(centers[:, 1], centers[:, 2], centers[:,3], c='black', s=200, alpha=0.5);
ax.set_xlabel('component 1')
ax.set_ylabel('component 2')
ax.set_zlabel('component 3')
plt.title('K-means clustering of political parties. K=10')

plt.show()

#loop over to get 13 plots:
for i in range(13):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced[:,i+1], reduced[:,i+2], reduced[:,i+3], c=y_clusters, cmap="hsv", marker='o', alpha=0.05)
    #centers = k_means.cluster_centers_
    #ax.scatter(centers[:, 1], centers[:, 2], centers[:,3], c='black', s=200, alpha=0.5);
    ax.set_xlabel('component '+ str(i+1))
    ax.set_ylabel('component ' + str(i+2))
    ax.set_zlabel('component ' + str(i+3))
    plt.title('K-means clustering of political parties. K=10')

    plt.show()
