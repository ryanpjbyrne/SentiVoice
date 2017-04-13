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
    #text=text.strip(".")
    #text=text.strip('"')
    tokens= nltk.word_tokenize(text)
    tokens=[token.strip() for token in tokens]
    return tokens

def tokenize(pos_rev, neg_rev):
    print "Hell"
    Bag=[]
    stop_words= set(nltk.corpus.stopwords.words('english'))
    stop_words.update(['rosemund','nick', 'gillian','elliot', 'read',
                       'readings','reading','writing', 'flynn', 'writings','novel', 'gone',
                       'girl','book', 'amazon', 'dunne', 'missouri',
                       'review','movie','spoiler','husband','wife','reader', 'plot', 'character','author',
                       'story', 'thriller','science','chapter','marriage','mystery','turner','hitchcock','fiction'
                       ,'girl','boy','train','writer',
                       'storyline','john','cancer','louie','page','atkinson',
                       'twilight','kindle','hazel','fifty','augustus',
                       'zamperini','seller','series','publisher','tablet','bella','john','twilight','june','porn','bdsm','book','movie'])
    for (words, sentiment) in pos_rev + neg_rev:
        words_filtered=[e.lower() for e in words if len(e)>3 and len(e)<10 if e not in list(punctuation) if e not in stop_words]
        #words_filtered=stem(words_filtered)
        Bag.append((words_filtered, sentiment))
    return Bag

def tokenize_text(text):
    extract=[]
    for words in text:
        filtered_words=[w.lower() for w in text if len(w)>=3 if w not in list(punctuation)]
        #filtered_words=stem(filtered_words)
    return filtered_words

def pos_tagger(tokens):
    tagged=[nltk.pos_tag(tokens)]    
    return tagged

def nouns_list(tags):
    tag_list=tags
    grammar= """
    JJ: {<JJ>}
    NNP: {<NNP>}
    VBG: {<VBG>}
    VBP: {<VBP>}
    NN:  {<NN>}
    
    
    """
    parser= nltk.RegexpParser(grammar)
    result=[parser.parse(tag) for tag in tag_list]
    return result
def Extract_Noun(chunck):
    result=[]
    for tree in chunck:
        chunks=[]
        noun_phrase=[subtree.leaves() for subtree in tree.subtrees() if subtree.label() == 'NN' or 'NNP' or 'NNS' or 'NNPS']
        for noun_phrases in noun_phrase:
            clean=[]
            for tag in noun_phrases:
                clean.append(tag[0])
            chunks.append(' '.join(clean))
    return chunks


def Extract_NP(chunck):
    result=[]
    for tree in chunck:
        chunks=[]
        noun_phrase=[subtree.leaves() for subtree in tree.subtrees() if subtree.label() == 'NN' or  subtree.label()== 'JJ' or subtree.label()== 'NNP' or subtree.label()== 'VBG' or subtree.label()=='VBP']
        for noun_phrases in noun_phrase:
            clean=[]
            for tag in noun_phrases:
                clean.append(tag[0])
            chunks.append(' '.join(clean))
    return chunks
def train_tagger(data):
    extract=[]
    for (words, sentiment) in data:
      hold=pos_tagger(words)
      nouns=nouns_list(hold)
      all_words=Extract_NP(nouns)
      extract.append((all_words,sentiment))
        
    return extract
def test_tagger(data):
    extract=[]
    pos_tags=pos_tagger(data)
    print pos_tags
    nouns=nouns_list(pos_tags)
    print nouns
    all_words=Extract_NP(nouns)
    return all_words  


def stem(text):
    stemmer=nltk.PorterStemmer()
    stemmed=[]
    for item in text:
        stemmed.append(stemmer.stem(item))
    return stemmed
def amazon_reviews():
  
    pos_rev=[]
    neg_rev=[]
    #neu_rev=[]
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
        no_training_examples=int(3000)
        print no_lines
        print no_training_examples
        half= (no_training_examples/2)
        halfed=(half/2)
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
        for line in lines[4000:(4000+halfed)]:
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
             
           
        
        f.close()
    return pos_rev, neg_rev,pos_test,neg_test
pos_rev, neg_rev,pos_test,neg_test =amazon_reviews()
#print pos_rev[0:2]
pos_rev=train_tagger(pos_rev)
neg_rev=train_tagger(neg_rev)
pos_test=train_tagger(pos_test)
neg_test=train_tagger(neg_test)
collect_train=tokenize(pos_rev, neg_rev)
collect_test=tokenize(pos_test, neg_test)
#collect=stop_words(collect)
#print collect[0:5]


word_features=get_word_features(get_words_in_bag(collect_train))

#word_test=get_word_features(get_words_in_bag(collect_test))


#print word_features

training_set = nltk.classify.apply_features(extract_features, collect_train)
#print "/n", training_set, "training set"
print len(training_set)
classifier = nltk.NaiveBayesClassifier.train(training_set) #REPLACE WITH SCI KIT MUCH FASTER
print classifier.show_most_informative_features(150)
word_features=get_word_features(get_words_in_bag(collect_test))
test_set= nltk.classify.apply_features(extract_features, collect_test)
print "%.3f" % nltk.classify.accuracy(classifier, test_set )
#1
entry= 'The vista was beautiful. I was in love with it.'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#2
entry= 'The Witcher was a boring book.'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#3
entry= 'I find sitting by the beach boring.'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#4
entry= 'Watching this anime is great. Can we watch some more?'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#6
entry= 'Gripping story right to the end.'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#7
entry= 'Overall very disappointing plot'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#8
entry= 'This article is a piece of crap.'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#9
entry= 'The game dragged and the story was flat.'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#10
entry= 'The Lego building was poorly constructed and looked awful.'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#11
entry= 'The humor in the movie was funny.'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#12
entry= 'I so nearly gave up on this book in the early chapters, as it was boring and repetitive. The storyline is weak and unbelievable.'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#13
entry= 'I hate talking about java.'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#14
entry= 'Just a brilliant book with a huge twist at the end making me want more.'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")
#15
entry='The Amazon Echo is a fantastic addition to any home. I highly recommend it.'
entry= clean_text(entry)
print entry
entry=test_tagger(entry)
print entry
entry=tokenize_text(entry)
print entry
result= classifier.classify(extract_features(entry))
print result
dist= classifier.prob_classify(extract_features((entry)))
print list(dist.samples())
print "pos"
print dist.prob("positive")
print "neu"
print dist.prob("neutral")
print "neg"
print dist.prob("negative")














