#do text similarities between clusters
import gensim
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from nltk.corpus import gutenberg

lemmatizer = WordNetLemmatizer()
sample = gutenberg.raw("bible-kjv.txt")
stopWords = set(stopwords.words('english'))

ex = sent_tokenize(sample)[:300]

#--------------      CLUSTERING      ----------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(ex)

model_dbscan = DBSCAN(eps = 1.1, min_samples = 2).fit(X)

labels = model_dbscan.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of DBSCAN clusters: %d' % n_clusters_)

unique_labels = set(labels)
cluster_list = list(unique_labels)

#ORGANIZE SENTENCES
#Create list of lists where each list represents a different cluster's sentences
# [ [ cluster 1 sentences]
#   [ cluster 2 sentences]
#   ....
#   [ cluster n senetences] ]

group_label = [[] for x in xrange(len(unique_labels)-1)]

noise_count = 0
for i in range(len(ex)):
    if labels[i] == -1:
        noise_count = noise_count + 1
        #does not add noise to a cluster group
        continue

    #appends the current sentence to the appropriate location
    #group_label is the list of lists
    #cluster_list.index(labels[i]) gets the location where sentence should be added
    group_label[cluster_list.index(labels[i])].append(ex[i])


tok_group_label = [[] for x in xrange(len(group_label))]

for i in range(len(group_label)):
    #tokenize each sentence in each cluster group, maintaining sentence order
    #filter out english stop words
    tok_group_label[i] = [[lemmatizer.lemmatize(w.lower()) for w in word_tokenize(text) if w not in stopWords] for text in group_label[i]]

#each sentence in the cluster is aggregated into a single array
cluster_grouping = [[] for x in xrange(len(group_label))]
for i in range(len(tok_group_label)):
    for j in range(len(tok_group_label[i])):
        cluster_grouping[i] = cluster_grouping[i] + tok_group_label[i][j]

#main_dict - dictionary of every word analyzed
main_dict = [item for sublist in tok_group_label for item in sublist]
main_dict = gensim.corpora.Dictionary(main_dict)

#Create corpus
#maps each word and word frequency to the dictionary
#corpus = []
corpus = [main_dict.doc2bow(cluster_group) for cluster_group in cluster_grouping]

#print(main_dict.token2id['god'])
print("Number of words in dictionary:",len(main_dict))
#for i in range(100):
#    print(i, main_dict[i])

tf_idf = gensim.models.TfidfModel(corpus)
sims = gensim.similarities.Similarity('/Users/Owner-PC/Documents/Coding/NLP/workdir/',tf_idf[corpus],num_features=len(main_dict))

s = 0
for i in corpus:
    s += len(i)
print(s)

#for i in range(len(corpus)):
#    sims.append(gensim.similarities.Similarity('/usr/workdir/',tf_idf[i][corpus[i]],num_features=len(main_dict)))

#print(sims)
#print(type(sims))

np.set_printoptions(precision=3, suppress=True)
query_doc = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize("The Above light Lights lights above above above and light from the sun gathered waters 1:9 fowl abundantly is from god") if w.lower() not in stopWords]
#print(query_doc)
query_doc_bow = main_dict.doc2bow(query_doc)
#print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)
sim_calculation = sims[query_doc_tf_idf]
print(sim_calculation)

query_doc = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize("2:12 As an offering of first-fruits ye may bring them unto the LORD; but they shall not come up for a sweet savour on the altar. 2:13 And every meal-offering of thine shalt thou season with salt; neither shalt thou suffer the salt of the covenant of thy God to be lacking from thy meal-offering; with all thy offerings thou shalt offer salt") if w.lower() not in stopWords]
query_doc_bow = main_dict.doc2bow(query_doc)
query_doc_tf_idf = tf_idf[query_doc_bow]
sim_calculation = sims[query_doc_tf_idf]

print(sim_calculation)
