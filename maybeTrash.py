
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
