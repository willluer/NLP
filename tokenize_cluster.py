import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score

lemmatizer = WordNetLemmatizer()

sun_dict = {'sunshine','sun','bright','hot', 'shine'}
rain_dict = {'rain', 'shower','sunshower', 'storm', 'rains'}
word_dict = sun_dict | rain_dict

ex = list(range(6))
token = list(range(6))

ex = ["the sunshine was strong and hot",
    "the hot sun is yellow",
    "i like sunshine",
    "There is rain in the storm",
    "rain shower during the day",
    "sunshower rain rain sun storm",
    "When it rain it pour",
    "I listen to rain drops during sunshower",
    "sun sun sun shine",
    "rain rain rain rain rain rain",
    "storm means to rain",
    "The dog is cute",
    "I play with the dog in the rain and the sun",
    "My cat and dog are not friends",
    "Rain is the name of my dog",
    "Cats and dogs do not get along",
    "Fetch dog",
    "Sun and rain and dog",
    "Go get a new dog",
    "Why is there a cloud during the sun",
    "Do not rain shower on me with my dog"
    "cat dog cat dog rain sun",
    "Cute dogs have a higher likability",
    "My dog name is hank",
    "Hello put on your sun screen unless there is rain"]

test = "It is going to rain and storm"

for i in range(6):
    token[i] = nltk.word_tokenize(ex[i])

print(token)
print()
print()

#---------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(ex)

true_k = 3
model_kmeans = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model_kmeans.fit(X)

print("Top terms per cluster:")
order_centroids = model_kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print


model_dbscan = DBSCAN(eps = 1.02, min_samples = 2).fit(X)

labels = model_dbscan.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of DBSCAN clusters: %d' % n_clusters_)
print('labels:')
print(labels)
print("\n")
print(model_dbscan.components_)
print("Prediction")

Y = vectorizer.transform(["storm storm storm storm"])
prediction = model_kmeans.predict(Y)
print(prediction)
prediction = model_dbscan.fit_predict(Y)
print(prediction)

Y = vectorizer.transform(["sun sun sun sun sun sun"])
prediction = model_kmeans.predict(Y)
print(prediction)
prediction = model_dbscan.fit_predict(Y)
print(prediction)
