#!/usr/bin/env python
# coding: utf-8

# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import re


# In[24]:


reviews_train = []
for line in open('F:\\software\\movie_data\\full_train.txt', 'r',encoding='utf8'):
    reviews_train.append(line.strip())
    
reviews_test = []
for line in open('F:\\software\\movie_data\\full_test.txt', 'r',encoding='utf8'):
    reviews_test.append(line.strip())


# In[25]:


REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)


# In[26]:


target = [1 if i < 12500 else 0 for i in range(25000)]


# In[27]:


stop_words = ['in', 'of', 'at', 'a', 'the']
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)


# In[29]:


classifier = LinearSVC(C=0.01)
classifier.fit(X, target)
print ("Final Accuracy: %s" % accuracy_score(target, classifier.predict(X_test)))


# In[48]:


a=[] #set of reviews
a.append("bad bad bad")
a.append("good bad bad")
a.append("good good good good a a ")


# In[49]:


def compute_sentiment(x):
    review_set_clean = preprocess_reviews(x)
    reviewset_vectorized = ngram_vectorizer.transform(review_set_clean)
    l=classifier.predict(reviewset_vectorized)
    print(l)
    k=0
    for i in l:
       k=k+i
    k=k/len(l)
    print("\nsentiment score is:",k)
    


# In[50]:


compute_sentiment(a)