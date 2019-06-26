###### SUPERVISED ML
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#### Nearest Neighbor
# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
X=reduced[:,1:]
y= d.loc[1:,"parfam"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=5, p=2, weights='uniform')
prediction = knn.predict(X_test)

#evaluation:
eval = prediction == y_test
sum(eval)/len(prediction) #success rate

#### TIME OUT: Save session
import dill
del context #this thing can't be saved in the session
filename="nearestneighbor_plot.pkl"
dill.dump_session(filename)

dill.load_session(filename) #when working again, load session with this


###cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X_test, y_test, cv=10, scoring='f1_macro')
scores.max()


###with the SVC Classifier
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X_train, y_train, cv=10)