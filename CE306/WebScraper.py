from bs4 import BeautifulSoup
import requests
import urllib2
import nltk
import re
from string import punctuation
from nltk import word_tokenize
from collections import Counter
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    lines=(line.strip() for line in text.splitlines())
    #break multi headlines
    breakmulti=(phrase.strip() for line in lines for phrase in line.split("   "))
    #drop blank lines
    text= '\n'.join(breakm for breakm in breakmulti if breakm)
    text=re.sub(r'\b\d+\b', ' ', text) #removes digits but keeps alphanumeric
    #removes standard url format
    url_pattern = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
    text = re.sub(url_pattern, ' ', text)
    text=text.lower()
    
    
    return text

def tokenize(text_tok):
    tokens= nltk.wordpunct_tokenize(text_tok)
    tokens=[token.strip() for token in tokens]
    #print(tokens)
    for t in tokens:
        filtered_words=[w for w in tokens if len(w)<20 if w not in list(punctuation)]
    return filtered_words
def stop_words(text_stop):

    stop_words= nltk.corpus.stopwords.words('english')

    for s in text_stop:
        #filters stop words
        removedstop=[w for w in text_stop if w not in stop_words]
    count=Counter(removedstop)
    #print count
    return removedstop

def stem(text):
    stemmer=nltk.PorterStemmer()
    stemmed=[]
    for item in text:
        stemmed.append(stemmer.stem(item))
    return stemmed

def pos_tagger(tokens):
    tagged=[nltk.pos_tag(tokens)]
        
    return tagged
    

#url=raw_input("Enter a website to extract information from:")
#r=requests.get("http://" + url)
#data=r.text
#print data
url="http://orb.essex.ac.uk/ce/ce306/syllabus.html"
page=urllib2.urlopen(url).read() #Query website and return the variable
soup=BeautifulSoup(page, 'html.parser')
#print soup.prettify()
[s.extract for s in soup(['style', 'script'])]
visible_text= soup.getText()
visible_text=clean_text(visible_text)
print visible_text
token_words=tokenize(visible_text)
#print(token_words)
token_stop=stop_words(token_words)
#print(token_stop)
stem_tokens=stem(token_stop)
print (stem_tokens)
pos_tags=pos_tagger(token_words)
#print (pos_tags)


vectorizer= TfidfVectorizer(min_df=1)
X= vectorizer.fit_transform(stem_tokens)
print X
#print(dict(zip(vectorizer.get_feature_names(),idf)))

#attempt input
str= 'CE306 CE706'
response=vectorizer.transform([str])
print response

