# for the neural net
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

import os
import urllib.request, json, ssl
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import cluster, preprocessing, decomposition
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


seed = 7
os.chdir("C:\\Users\\trmcdade\\OneDrive\\Laptop\\Professional\\SICSS RTI\\Party Manifesto")

## GET DATA
#bypass SSL verification
context = ssl._create_unverified_context()
with urllib.request.urlopen("https://manifesto-project.wzb.eu/tools/api_get_core.json?api_key=0a3513928be002d4784e16726d50b735&key=MPDS2018b", context=context) as url:
        cmp_test = json.loads(url.read().decode())
#create index col
index = list(range(len(cmp_test)))
#turn imported data into pd dataframe
d = pd.DataFrame(data = cmp_test, columns = cmp_test[0], index = index)
d = d.iloc[1:d.shape[0]]
# should we get rid of 999?

## UNSUPERVISED ML

## Algorithm 1: K-means clustering
# Preprocessing:
# turn labels into numeric values s.t. k_means takes them
# testdf = testdf.fillna(value=999) #replace all NAs with 999

#result = isolate a target series
# re-code the output variable so it works in the neural net
unique_outputs = pd.DataFrame([(x,i) for i,x in enumerate(d['parfam'].unique())])
unique_outputs.columns = ['parfam', 'target']
# add in new dv column to the dataframe
d = d.merge(unique_outputs, how = 'left', left_on = 'parfam', right_on = 'parfam')

# Do PCA to reduce the dimensionality
# See how many principal components we should use
# create PCA dataframe
# modify the data set so PCA can be run on it
pca_data = d
pca_data = pca_data.iloc[:,10:(len(pca_data.columns) - 1)]
# sk.preprocessing.normalize(pca_data, axis = 0)
le = preprocessing.LabelEncoder()
for column_name in pca_data.columns:
       if pca_data[column_name].dtype == object:
           pca_data[column_name] = le.fit_transform(pca_data[column_name].astype(str))
       else:
           pass

matrix = [[0] * 2 for i in range(len(pca_data.columns) - 1)]
pca_output = pd.DataFrame(matrix)
pca_output.columns = ['num_components', '% explained']
pca_output['% explained'] = pca_output['% explained'].astype(float)

# see which principal components have diminishing marginal returns
for i in range(1, len(pca_data.columns)):
    pca = PCA(n_components = i)
    pca.fit(pca_data)
    X_pca = pca.transform(pca_data)
    pca_output.iloc[i - 1, 0] = i
    pca_output.iloc[i - 1, 1] = sum(pca.explained_variance_ratio_)
plt.scatter(pca_output.iloc[:,0], pca_output.iloc[:,1], alpha=0.2)
plt.axvline(x=40, c = 'red')
plt.xlabel('nth Principal Component')
plt.ylabel('% Variance Explained')
plt.title("Explained Variance for the nth Principal Component")
plt.text(42, 0.9, '97.6%')
plt.show()
plt.savefig('pc_cutoff.png', dpi = 1000, bbox_inches = 'tight')

## do PCA: Choose 40 dimensions from above
ncomponents = 40
pca = PCA(n_components = ncomponents)
pca.fit(pca_data)
X_pca = pca.transform(pca_data)
# sum(pca.explained_variance_ratio_) # is the % of variance explained
# retreive the principal components for use in the clustering algorithm
transformed_df = pd.DataFrame(X_pca)

## Build the principal component/original feature heat map
# weight matrix
weight_df = pd.DataFrame(pca.components_,
                         columns = pca_data.columns,
                         index = ["PC-" + str(i) for i in range(1, len(pca.components_) + 1)])
# plot the weight matrix as a heat map
fig, ax = plt.subplots(1, 1, figsize = (16,6))
im = ax.imshow(weight_df,
               # interpolation = 'kaiser',
               cmap = 'Spectral')
ax.set_xticks(np.arange(len(weight_df.columns)))
ax.set_yticks(np.arange(len(weight_df.index)))
ax.set_xticklabels(weight_df.columns)
ax.set_yticklabels(weight_df.index)
ax.xaxis.label.set_fontsize = ax.yaxis.label.set_fontsize = 5
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax.set_title("Feature Composition of First 40 Principal Components")
fig.tight_layout()
plt.savefig('pc_heatmap.png', dpi = 1000, bbox_inches = 'tight')
plt.show()


## Now, create the NN
# create the target variable matrix for the nn
target_var = np.array(d['target'])
target = np_utils.to_categorical(target_var)

## K means
d['target'].unique()
k_means = cluster.KMeans(n_clusters=len(d['target'].unique())) #follow the codebook. assume some 10 groups
kmeans_df = pd.concat([transformed_df, d['target']], axis = 1).dropna()
k_means.fit(kmeans_df)
#check the output groups
clustered = k_means.labels_
print(clustered)
print(len(clustered)) #why are we getting different N?
#extract output
y = k_means.predict(kmeans_df)


# run a train/test split nn
X_train, X_test, y_train, y_test = train_test_split(
                                                    transformed_df,
                                                    target,
                                                    test_size=0.33,
                                                    random_state=42)

class_names = list(d['parfam'].unique())

#scale values
X_train = sk.preprocessing.normalize(X_train)
X_test = sk.preprocessing.normalize(X_test)

nnodes_layer1 = 256
nnodes_layer2 = 256
nnodes_layer3 = 128
nnodes_layer4 = 32
model = keras.Sequential([
    # keras.layers.Flatten(input_shape=(X_train.shape[0], X_train.shape[1])),
    keras.layers.Dense(nnodes_layer1, input_dim = 40, activation=tf.nn.relu),
    keras.layers.Dense(nnodes_layer2, input_dim = 40, activation=tf.nn.relu),
    keras.layers.Dense(nnodes_layer3, input_dim = 40, activation=tf.nn.relu),
    keras.layers.Dense(nnodes_layer4, input_dim = 40, activation=tf.nn.relu),
    keras.layers.Dense(len(d['parfam'].unique()), activation=tf.nn.softmax)
])

keras.optimizers.Adam(lr = 0.05)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 10)

test_loss, test_acc = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
print('Test accuracy:', test_acc)


### Run the nn with kfold Cross Validation
def baseline_model():
    #create model
    model = keras.Sequential([
        # keras.layers.Flatten(input_shape=(X_train.shape[0], X_train.shape[1])),
        keras.layers.Dense(128, input_dim = 40, activation=tf.nn.relu),
        keras.layers.Dense(len(d['parfam'].unique()), activation=tf.nn.softmax)
    ])
    #compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs = 200,
                            batch_size = 5,
                            verbose = 0)
kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)
results = cross_val_score(estimator, transformed_df, target, cv = kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

### below here: figure out visualization
def plot_image(i, predictions_array, true_label):
  predictions_array = predictions_array[i]
  true_label = np.argmax(true_label[i])
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  predicted_label = np.argmax(predictions_array)
  true_label = np.argmax(true_label)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  thisplot = plt.bar(range(len(d['parfam'].unique())), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

matches = notmatches = 0
for i in range(y_test.shape[0]):
    # plt.figure()
    # plot_image(i, predictions, y_test)
    # plt.show()
    if np.argmax(predictions[i]) == np.argmax(y_test):
        matches += 1
    else:
        notmatches += 1

df = pd.DataFrame()
df.at[0, 'matches'] = matches
df.at[0, 'non-matches'] = notmatches
df = {'matches':matches, 'non-matches':notmatches}
fig, axs = plt.subplots()
axs.bar(list(df.keys()), list(df.values()))
fig.suptitle('Categorical Plotting')

fig1, ax1 = plt.subplots()
ax1.pie(df.values(), labels = df.keys(),
        autopct = '%1.1f%%', shadow = True, startangle = 90)
ax1.axis('equal')
plt.show()


## After here, it's not quite useful.

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, y_test, X_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, y_test)
plt.show()

# Grab an image from the test dataset
img = X_test[0]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, y_test)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])
