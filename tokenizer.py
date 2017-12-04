import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

EXAMPLE_TEXT = "Hello Hello hello heLlo Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard. eriod, space, and then a capital letter. The problem is that things like Mr. Smith would cause you trouble, and many other things. Splitting by word is also a challenge, especially when considering things like concatenations like we and are to we're. NLTK is going to go ahead and just save you a ton of time with this seemingly simple, yet very complex, operation. here are a few things to note here. First, notice that punctuation is treated as a separate token. Also, notice the separation of the word  into and  Finally, notice that pinkish-blue is indeed treated like the it was meant to be turned into. Pretty cool!Now, looking at these tokenized words, we have to begin thinking about what our next step might be. We start to ponder about how might we derive meaning by looking at these words. We can clearly think of ways to put valut"

tokens = nltk.word_tokenize(EXAMPLE_TEXT)
print(tokens)
