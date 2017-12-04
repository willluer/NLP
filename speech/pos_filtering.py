import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

filename = "dog_wiki.txt"
file_total = open(filename, "r")
file_read = "" + file_total.read()

sent_tok = sent_tokenize(file_read)
print(type(sent_tok))
sent_tok = sent_tok[:20]
print(sent_tok)
#print(sent_tok)
word_tok = word_tokenize(sent_tok)
#print(file_tok_sample)
