import nltk
import random
from nltk.tokenize import sent_tokenize, word_tokenize

HELLO_TERMS = ["hello", "hi","yo","sup mothafucka","greetings","hey"]
BYE_TERMS = ["bye", "see ya", "goodbye", "peace","later"]
DID_NOT_COMPREHEND = ["the fuck you say?", "say something else poser", "eat ass"]

def simple_response(sentence):
    for word in word_tokenize(sentence):
        if word.lower() in HELLO_TERMS:
            response = random.choice(HELLO_TERMS)
        elif word.lower() in BYE_TERMS:
            response = random.choice(BYE_TERMS)
        else:
            response = random.choice(DID_NOT_COMPREHEND)
    return response


#while True:

    #var = raw_input("Say hello to Wally... \n  Input: ")
    #response = "  Response: " + simple_response(var)
    #sentence = raw_input("Input a sentence to be tagged... \n  Input: ")
sentence = "hey man I like to eat red apples on the beach during the dark night"
print(sentence)
print(nltk.pos_tag(word_tokenize(sentence)))
