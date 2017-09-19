import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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
    "Fetch boy"]

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
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = vectorizer.transform(["storm storm storm storm"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["sun sun sun sun sun sun"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["Do you like my dog"])
prediction = model.predict(Y)
print(prediction)
