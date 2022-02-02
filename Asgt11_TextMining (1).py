#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import requests
from bs4 import BeautifulSoup

#request url acess and get html page content
url = 'https://www.makeupalley.com/product/showreview.asp/ItemId=140186/Baby-Lips-Lip-Balm/Maybelline-New-York/Lip-Treatments'
page = requests.get(url)
htmlcontent = page.content
print(htmlcontent)


#parsing the html ontent with Beautiful Soup
soup = BeautifulSoup(htmlcontent, 'html.parser')
print(soup.prettify)

#Get all reviews from page
reviews = soup.find_all('p')
print(reviews)


# In[2]:


print(soup.find_all('p', class_="review-text-readmore"))


# In[3]:


a = soup.find_all('p', class_='review-text-readmore')


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import string
import spacy
from matplotlib.pyplot import imread
from wordcloud import wordcloud


# In[5]:


import nltk
nltk.download('punkt')


# In[6]:


from nltk import tokenize
sentences = tokenize.sent_tokenize(''.join(str(a)))
sentences


# In[7]:


df_sent = pd.DataFrame(sentences, columns = ['sentence'])
df_sent


# In[8]:


#Sentiment analysis
afinn = pd.read_csv('C:/Users/17pol/Downloads/Afinn.csv', sep=',', encoding='latin-1')
afinn.shape


# In[9]:


affinity_scores = afinn.set_index('word')['value'].to_dict()


# In[10]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[11]:


affinity_scores


# In[12]:


#Custom function :score each word in a sentence in lemmatised form, 
#but calculate the score for the whole original sentence.
nlp = spacy.load('en_core_web_sm')
sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score


# In[13]:


# test that it works
calculate_sentiment(text = 'excellent')


# In[14]:


df_sent['sentiment_value'] = df_sent['sentence'].apply(calculate_sentiment)


# In[15]:


# how many words are in the sentence?
df_sent['word_count'] = df_sent['sentence'].str.split().apply(len)
df_sent['word_count'].head(10)


# In[16]:


df_sent.sort_values(by='sentiment_value').tail(10)


# In[17]:


# Sentiment score of the whole review
df_sent['sentiment_value'].describe()


# In[18]:


# Sentiment score of the whole review
df_sent[df_sent['sentiment_value']<=0].head()


# In[20]:


df_sent[df_sent['sentiment_value']==5].head()


# In[21]:


df_sent['index']=range(0,len(df_sent))


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(df_sent['sentiment_value'])


# In[23]:


plt.figure(figsize=(15, 10))
sns.lineplot(y='sentiment_value',x='index',data=df_sent)


# In[24]:


# Maximum sentiment value
df_sent['sentiment_value'].max()


# In[29]:


# review which is having max sentiment value
df_sent[df_sent['sentiment_value']==5]


# In[26]:


# Full review at index 105
df_sent['sentence'][105]


# In[27]:


# minimum sentiment value
df_sent['sentiment_value'].min()


# In[28]:


# review which is having min sentiment value
df_sent[df_sent['sentiment_value']==-6]


# In[30]:


# Full review at index 139
df_sent['sentence'][139]


# In[31]:


df_sent.plot.scatter(x='word_count', y='sentiment_value', figsize=(8,8), title='Sentence sentiment value to sentence word count')


# In[34]:


df_sent['Sentiment_Class'] = pd.cut(x=df_sent['sentiment_value'],bins=[-8, -1, 0, 6], 
                                    labels=['Negative','Neutral','Positive'], right = True)


# In[35]:


df_sent.sample(10)


# In[36]:


sns.countplot(x = 'Sentiment_Class', data = df_sent)


# In[37]:


df_sent['Sentiment_Class'].value_counts()


# In[ ]:




