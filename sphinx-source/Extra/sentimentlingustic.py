import nltk,re,os
import numpy as np
import random
from io import open

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words
    
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

PosTrain='rt-polarity-pos.txt'
NegTrain='rt-polarity-neg.txt'
pos_tweets=[]
neg_tweets=[]
tweets = []

def amazon_reviews():
  
    pos_tweets=[]
    neg_tweets=[]
    datafolder = '/home/ryan/Desktop/sphinx-source/amazon/'
    files = os.listdir(datafolder)
    Y_train, Y_test, X_train, X_test,  = [], [], [], []
    for file in files:
        print file
        f = open(datafolder + file, 'r', encoding="utf8")
        label = file
        lines = f.readlines()
        no_lines = len(lines)
        no_training_examples = int(0.7*no_lines)
        print no_lines
        print no_training_examples
        for line in lines[:no_training_examples]:
            if label=="pos":
                posWords=line
                posWords=[posWords,"positive"]
                pos_tweets.append(posWords)
            if label=="neg":
                negWords=line
                negWords = [negWords, 'negative']
                neg_tweets.append(negWords)
           
        
        f.close()
    return pos_tweets, neg_tweets
pos_tweets, neg_tweets=amazon_reviews()



'''
with open(PosTrain, 'r') as posSentences:
    for i in posSentences:
        posWords = i
        posWords = [posWords, 'positive']
        pos_tweets.append(posWords)
with open(NegTrain, 'r') as negSentences:
    for i in negSentences:
        negWords = i
        negWords = [negWords, 'negative']
        neg_tweets.append(negWords)
print (pos_tweets[1])
print(neg_tweets[1])
'''
for (words, sentiment) in pos_tweets[0:2000] + neg_tweets[0:2000]:
    words_filtered=[e.lower() for e in words.split() if len(e)>3]
    tweets.append((words_filtered, sentiment))
    
#print tweets[0:5]

word_features=get_word_features(get_words_in_tweets(tweets))


#print word_features

training_set = nltk.classify.apply_features(extract_features, tweets)
#print "/n", training_set, "training set"
classifier = nltk.NaiveBayesClassifier.train(training_set) #REPLACE WITH SCI KIT MUCH FASTER
input= 'I like playing tennis'
print classifier.classify(extract_features(input.split()))


