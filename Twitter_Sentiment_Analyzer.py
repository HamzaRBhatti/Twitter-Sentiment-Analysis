from textblob import TextBlob
from random import shuffle
from nltk.corpus import twitter_samples
from textblob.classifiers import NaiveBayesClassifier

#
#
# text="im very hungry"
# analysis=TextBlob(text)
# print(text)
# print(analysis.sentiment)
#
# if analysis.sentiment[0]>0:
#     print ("Positive")
# elif analysis.sentiment[0]<0:
#     print ("Negative")
# else:
#    print ("Neutral")



print(twitter_samples.fileids())
'''
Output:

['negative_tweets.json', 'positive_tweets.json', 'tweets.20150430-223406.json']
'''

pos_tweets = twitter_samples.strings('positive_tweets.json')
print(len(pos_tweets))  # Output: 5000

neg_tweets = twitter_samples.strings('negative_tweets.json')
print(len(neg_tweets))  # Output: 5000

# all_tweets = twitter_samples.strings('tweets.20150430-223406.json')
# print (len(all_tweets)) # Output: 20000

# positive tweets words list
pos_tweets_set = []
for tweet in pos_tweets:
    pos_tweets_set.append((tweet, 'pos'))

# negative tweets words list
neg_tweets_set = []
for tweet in neg_tweets:
    neg_tweets_set.append((tweet, 'neg'))

print(len(pos_tweets_set), len(neg_tweets_set))  # Output: (5000, 5000)
print(pos_tweets_set)
print("------------------------------")
print(neg_tweets_set)

# radomize pos_reviews_set and neg_reviews_set
# doing so will output different accuracy result everytime we run the program


shuffle(pos_tweets_set)
shuffle(neg_tweets_set)

# test set = 200 tweets (100 positive + 100 negative)
# train set = 400 tweets (200 positive + 200 negative)
test_set = pos_tweets_set[:100] + neg_tweets_set[:100]
train_set = pos_tweets_set[100:300] + neg_tweets_set[100:300]

print(len(test_set), len(train_set))  # Output: (200, 400)


# train classifier
classifier = NaiveBayesClassifier(train_set)

# calculate accuracy
accuracy = classifier.accuracy(test_set)
print(accuracy)  # Output: 0.715


# show most frequently occurring words
print (classifier.show_informative_features(10))

text = "It was a wonderful movie. I liked it very much."
print (classifier.classify(text)) # Output: pos





