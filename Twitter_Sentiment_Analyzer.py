from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk import classify
from nltk import NaiveBayesClassifier
import tkinter as tk
import string
import re as regex
from random import shuffle


stop_words=stopwords.words('english')
stemmer = PorterStemmer()


#Fetching all the tweets dataset 
Positive_Tweets=twitter_samples.strings('positive_tweets.json')
print(Positive_Tweets)
print(len(Positive_Tweets))

Negative_Tweets=twitter_samples.strings('negative_tweets.json')
print(Negative_Tweets)
print(len(Negative_Tweets))


# for tweet in Positive_Tweets:
#     print(tweet_tokenizer.tokenize(tweet))

# Happy Emoticons to remove them from each tweet
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])

# Sad Emoticons to remove them from each tweet
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])


# all emoticons
emoticons=emoticons_happy.union(emoticons_sad)


def clean_tweets(tweet):
    tweet = regex.sub(r'\$\w*', '', tweet)

    tweet = regex.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    tweet = regex.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    tweet = regex.sub(r'#', '', tweet)

    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stop_words and  # remove stopwords
                word not in emoticons and  # remove emoticons
                word not in string.punctuation):  # remove punctuation

            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


# custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"
# print(clean_tweets(custom_tweet))

# feature extractor function
def bag_of_words(tweet):
    words=clean_tweets(tweet)
    words_dictionary=dict([word, True] for word in words)
    return words_dictionary


# custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"
# print(bag_of_words(custom_tweet))

#positive tweet feature vector
Positive_Tweets_set=[]
for tweet in Positive_Tweets:
    Positive_Tweets_set.append((bag_of_words(tweet),'positive'))

# print(Positive_Tweets_set)

#Negative teet feature vector
neg_tweet_set=[]
for tweet in Negative_Tweets:
    neg_tweet_set.append((bag_of_words(tweet),'negative'))

# print(len(Positive_Tweets_set),len(neg_tweet_set))

shuffle(Positive_Tweets_set)
shuffle(neg_tweet_set)

test_set= Positive_Tweets_set[:1000] + neg_tweet_set[:1000]
train_set=Positive_Tweets_set[:3000] + neg_tweet_set[:3000]

print(len(test_set),len(train_set))


classifier = NaiveBayesClassifier.train(train_set)

accuracy=classify.accuracy(classifier,test_set)
print(accuracy)


# check()
root = tk.Tk(className='Twitter Sentiment Analyzer')
root.geometry("700x400")
root.config(bg="#d3e9fe")



label = tk.Label(root, text="Twitter Sentiment Analysis",bg="#02557d",fg="white",bd=5,height=2,width=100,font='Helvetica 18 bold')
# label.grid(row=0, column=1)
label.pack()

textExample = tk.Text(root, height=2,width=50)
# textExample.grid(row=3, column=4)
textExample.pack(pady=15)

def  check():
# print(classifier.mostinformative_features(10))
    custom_tweet =  textExample.get("1.0", "end")
    print(custom_tweet)
    custom_tweet_set = bag_of_words(custom_tweet)
    classi = classifier.classify(custom_tweet_set)  # Output: pos
    label.config(text=classi)
    print("The result is:",classi,"itive")
    return classi
# Positive tweet correctly classified as positive

btnRead = tk.Button(root, height=1, width=14,bg="#15acee", text="Click to Analyze", command=check)
btnRead.pack()



label = tk.Label(root, text="",font='Helvetica 18 bold',width=40)
label.pack(pady=10)

label1 = tk.Label(root, text="Twitter sentiment analyzer predicts the tweets and classifies as positve or negative ",font='Helvetica 12 bold',height=7,width=100,bg="#02557d",fg="white")
label1.pack(pady=6)

# Label3.grid(row=3, column=0)
# Entry3.grid(row=3, column=1, sticky="ew")


label2 = tk.Label(root, text=" Developed By Hamza Bhatti ",font='Helvetica 12 bold',height=2,width=100,bg="#d3e9fe",fg="Black")
label2.pack()

root.mainloop()