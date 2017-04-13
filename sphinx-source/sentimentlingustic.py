import nltk,re,os
import numpy as np
import random
from io import open
from string import punctuation

def get_words_in_bag(bag):
    all_words = []
    for (words, sentiment) in bag:
      all_words.extend(words)
    return all_words
    
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    #print wordlist.most_common(200)
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def clean_text(text):
    lines=(line.strip() for line in text.splitlines())
    #break multi headlines
    breakmulti=(phrase.strip() for line in lines for phrase in line.split("   "))
    #drop blank lines
    text= '\n'.join(breakm for breakm in breakmulti if breakm)
    text=text.lower()
    tokens= nltk.wordpunct_tokenize(text)
    tokens=[token.strip() for token in tokens]
    return tokens

def tokenize(pos_rev, neg_rev, neu_rev):
    Bag=[]
    stop_words= set(nltk.corpus.stopwords.words('english'))
    stop_words.update(['rosemund','nick', 'gillian','elliot', 'read',
                       'readings','reading','writing', 'flynn', 'writings','novel', 'gone',
                       'girl','book', 'amazon', 'dunne', 'missouri','star',
                       'review','movie','spoiler','husband','wife','reader', 'plot', 'character','author',
                       'story', 'thriller','science','chapter','marriage','mystery','turner','hitchcock','fiction'
                       ,'girl','boy','train','writer',
                       'storyline','john','cancer','louie','page','atkinson',
                       'twilight','kindle','hazel','fifty','augustus',
                       'zamperini','seller','series','publisher'])
    for (words, sentiment) in pos_rev[0:2000] + neg_rev[0:2000]+neu_rev[0:2000]:
        words_filtered=[e.lower() for e in words if len(e)>3 and len(e)<10 if e not in list(punctuation) if e not in stop_words]
        hold=pos_tagger(words_filtered)
        nouns=nouns_list(hold)
        all_words=Extract_NP(nouns)
        all_words=stem(all_words)
        Bag.append((all_words, sentiment))
    return Bag

def pos_tagger(tokens):
    tagged=[nltk.pos_tag(tokens)]    
    return tagged

def nouns_list(tags):
    tag_list=tags
    grammar= """
    NP: {<DT>?<JJ>*<NN>}
    """
    parser= nltk.RegexpParser(grammar)
    result=[parser.parse(tag) for tag in tag_list]
    return result

def Extract_NP(chunck):
    result=[]
    for tree in chunck:
        chunks=[]
        noun_phrase=[subtree.leaves() for subtree in tree.subtrees() if subtree.label() == 'NP']
        for noun_phrases in noun_phrase:
            clean=[]
            for tag in noun_phrases:
                clean.append(tag[0])
            chunks.append(' '.join(clean))
    return chunks

def stem(text):
    stemmer=nltk.PorterStemmer()
    stemmed=[]
    for item in text:
        stemmed.append(stemmer.stem(item))
    return stemmed
def amazon_reviews():
  
    pos_rev=[]
    neg_rev=[]
    neu_rev=[]
    pos_test=[]
    neg_test=[]
    neu_test=[]
    datafolder = '/home/ryan/Desktop/sphinx-source/amazon/'
    files = os.listdir(datafolder)
    for file in files:
        print file
        f = open(datafolder + file, 'r', encoding="utf8")
        label = file
        lines = f.readlines()
        no_lines = len(lines)
        no_training_examples=int(360)
        print no_lines
        print no_training_examples
        half= (no_training_examples/2)
        for line in lines[40000:(40000+half)]:
            if label=="pos":
                posWords=line
                posWords=clean_text(posWords)
                posWords=[posWords,"positive"]
                pos_rev.append(posWords)
        for line in lines[(no_lines-3000-half):(no_lines-3000)]:        
            if label=="neg":
                negWords=line
                negWords=clean_text(negWords)
                negWords = [negWords, 'negative']
                neg_rev.append(negWords)
        for line in lines[no_training_examples:(no_training_examples+half)]:    
            if label=="neu":
                neuWords=line
                neuWords=clean_text(neuWords)
                neuWords=[neuWords, 'neutral']
                neu_rev.append(neuWords)
        for line in lines[40000:(40000+half)]:
             if label=="pos":
                posWords=line
                posWords=clean_text(posWords)
                posWords=[posWords,"positive"]
                pos_test.append(posWords)
             if label=="neg":
                negWords=line
                negWords=clean_text(negWords)
                negWords = [negWords, 'negative']
                neg_test.append(negWords)
             if label=="neu":
                neuWords=line
                neuWords=clean_text(neuWords)
                neuWords=[neuWords, 'neutral']
                neu_test.append(neuWords)
           
        
        f.close()
    return pos_rev, neg_rev, neu_rev,pos_test,neg_test,neu_test
pos_rev, neg_rev, neu_rev,pos_test,neg_test,neu_test=amazon_reviews()
#print pos_rev[0:2]

collect_train=tokenize(pos_rev, neg_rev, neu_rev)
collect_test=tokenize(pos_test, neg_test, neu_test)
#collect=stop_words(collect)
#print collect[0:5]


word_features=get_word_features(get_words_in_bag(collect_train))

#word_test=get_word_features(get_words_in_bag(collect_test))


#print word_features

training_set = nltk.classify.apply_features(extract_features, collect_train)
#print "/n", training_set, "training set"
classifier = nltk.NaiveBayesClassifier.train(training_set) #REPLACE WITH SCI KIT MUCH FASTER
print classifier.show_most_informative_features(150)
word_features=get_word_features(get_words_in_bag(collect_test))
test_set= nltk.classify.apply_features(extract_features, collect_test)
input= 'I hate reading gross articles about water.'
result= classifier.classify(extract_features(input.split()))
print classifier.classify(extract_features(input.split()))
print "%.3f" % nltk.classify.accuracy(classifier, test_set )
dist= classifier.prob_classify(extract_features(input.split()))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#print extract_features(input.split())



