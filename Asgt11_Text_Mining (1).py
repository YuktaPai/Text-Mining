#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install spacy


# In[3]:


pip install wordcloud


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud
from matplotlib.pyplot import imread
import string
import spacy


# ### Data

# In[11]:


df = pd.read_csv('C:/Users/17pol/Downloads/Elon_musk.csv', encoding = 'latin1', error_bad_lines=False)
df.head()


# ### Data Preprocessing

# In[12]:


df.shape


# In[13]:


df.info()


# In[14]:


df.isna().sum()


# In[15]:


#dropping  additional index column
df = df.drop('Unnamed: 0', axis = 1)


# In[16]:


#Renaming Text column
df = df.rename({'Text':'Tweets'}, axis = 1)
df.head()


# In[17]:


#removing both leading and trailing characters such as spaces in tweets
df = [x.strip() for x in df.Tweets]


# In[18]:


# removes empty strings, because they are considered in Python as False
df = [x for x in df if x]


# In[19]:


df[0:10]


# In[20]:


import nltk
nltk.download('punkt')


# In[22]:


from nltk import tokenize
sentences = tokenize.sent_tokenize(''.join(df))
sentences


# In[25]:


df_sent = pd.DataFrame(sentences, columns = ['sentence'])
df_sent


# In[27]:


#Sentiment analysis
afinn = pd.read_csv('C:/Users/17pol/Downloads/Afinn.csv', sep=',', encoding='latin-1')
afinn.shape


# In[29]:


afinn.head()


# In[30]:


affinity_scores = afinn.set_index('word')['value'].to_dict()


# In[33]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[68]:


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


# In[69]:


# test that it works
calculate_sentiment(text = 'excellent')


# In[38]:


df_sent['sentiment_value'] = df_sent['sentence'].apply(calculate_sentiment)


# In[40]:


# how many words are in the sentence?
df_sent['word_count'] = df_sent['sentence'].str.split().apply(len)
df_sent['word_count'].head(10)


# In[42]:


df_sent.sort_values(by='sentiment_value').tail(10)


# In[43]:


# Sentiment score of the whole review
df_sent['sentiment_value'].describe()


# In[44]:


# Sentiment score of the whole review
df_sent[df_sent['sentiment_value']<=0].head()


# In[46]:


df_sent[df_sent['sentiment_value']>=10].head()


# In[47]:


df_sent['index']=range(0,len(df_sent))


# In[48]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(df_sent['sentiment_value'])


# In[50]:


plt.figure(figsize=(15, 10))
sns.lineplot(y='sentiment_value',x='index',data=df_sent)


# In[52]:


# Maximum sentiment value
df_sent['sentiment_value'].max()


# In[53]:


# tweet which is having max sentiment value
df_sent[df_sent['sentiment_value']==16]


# In[58]:


# Full tweet at index 99
df_sent['sentence'][99]


# In[56]:


# minimum sentiment value
df_sent['sentiment_value'].min()


# In[57]:


# tweet which is having min sentiment value
df_sent[df_sent['sentiment_value']==-7]


# In[60]:


# Full tweet at index 59
df_sent['sentence'][59]


# In[61]:


# Full tweet at index 824
df_sent['sentence'][824]


# In[62]:


df_sent.plot.scatter(x='word_count', y='sentiment_value', figsize=(8,8), title='Sentence sentiment value to sentence word count')


# In[63]:


df_sent['Sentiment_Class'] = pd.cut(x=df_sent['sentiment_value'],bins=[-8, -1, 0, 17], 
                                    labels=['Negative','Neutral','Positive'], right = True)


# In[64]:


df_sent.sample(10)


# In[66]:


sns.countplot(x = 'Sentiment_Class', data = df_sent)


# In[67]:


df_sent['Sentiment_Class'].value_counts()

