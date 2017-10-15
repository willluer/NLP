#do text similarities between clusters

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score
from nltk.corpus import gutenberg

sample = gutenberg.raw("bible-kjv.txt")

ex = sent_tokenize(sample)

ex = ex[:1000]

#---------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(ex)

true_k = 3
#model_kmeans = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
#model_kmeans.fit(X)

#print("Top terms per cluster:")
#order_centroids = model_kmeans.cluster_centers_.argsort()[:, ::-1]
#terms = vectorizer.get_feature_names()
#for i in range(true_k):
#    print("Cluster %d:" % i),
#    for ind in order_centroids[i, :10]:
#        print(' %s' % terms[ind]),
#    print


model_dbscan = DBSCAN(eps = 1.1, min_samples = 2).fit(X)

labels = model_dbscan.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of DBSCAN clusters: %d' % n_clusters_)
print("\n")
#print("Prediction")
unique_labels = set(labels)
group_0 = []
for i in range(len(ex)):
    if labels[i] == 0:
        group_0.append(ex[i])

#group_0 = [w for w in ex if w in labels is 0]
print(group_0)
Y = vectorizer.transform(["rain"])
#prediction = model_kmeans.predict(Y)
#print(prediction)
prediction = model_dbscan.fit_predict(Y)
print(prediction)

Y = vectorizer.transform(["Let the sun makith light for the people"])
#prediction = model_kmeans.predict(Y)
#print(prediction)
prediction = model_dbscan.fit_predict(Y)
print(prediction)
