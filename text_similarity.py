#do text similarities between clusters
import gensim
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from nltk.corpus import gutenberg

sample = gutenberg.raw("bible-kjv.txt")
ex = sent_tokenize(sample)
ex = ex[:100]

#---------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(ex)

model_dbscan = DBSCAN(eps = 1.1, min_samples = 2).fit(X)

labels = model_dbscan.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of DBSCAN clusters: %d' % n_clusters_)
print("\n")
#print("Prediction")

unique_labels = set(labels)
cluster_list = list(unique_labels)

#ORGANIZE SENTENCES
#Create list of lists where each list represents a different cluster
group_label = [[] for x in xrange(len(unique_labels)-1)]

noise_count = 0
for i in range(len(ex)):
    if labels[i] == -1:
        noise_count = noise_count + 1
        #To print the 'noise', uncomment line below
        #print ex[i]
        continue
    group_label[cluster_list.index(labels[i])].append(ex[i])

tok_groups = [[] for x in xrange(len(group_label))]

for i in range(len(group_label)):
    tok_groups[i] = [[w.lower() for w in word_tokenize(text)] for text in group_label[i]]

main_dict = [item for sublist in tok_groups for item in sublist]
main_dict = gensim.corpora.Dictionary(main_dict)

corpus = []
for i in range(len(tok_groups)):
    corpus.append([main_dict.doc2bow(entry) for entry in tok_groups[i]])
#print(corpus)

#print(main_dict.token2id['god'])
print("Number of words in dictionary:",len(main_dict))
#for i in range(100):
#    print(i, main_dict[i])

tf_idf = gensim.models.TfidfModel(corpus[0])
#print(tf_idf)
s = 0
for i in corpus:
    s += len(i)
#print(s)

sims = gensim.similarities.Similarity('/usr/workdir/',tf_idf[corpus[0]],num_features=len(main_dict))
print("corpus[0] = ", corpus[0])
print(sims)
print(type(sims))

query_doc = [w.lower() for w in word_tokenize("The Above above above above and light from the sun is from god")]
print(query_doc)
query_doc_bow = main_dict.doc2bow(query_doc)
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)
