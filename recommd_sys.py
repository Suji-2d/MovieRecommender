#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import nltk
import re
from nltk.stem.porter import PorterStemmer
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
from sklearn.metrics.pairwise import cosine_similarity


# In[62]:


k_drama_df= pd.read_csv("D:\Developement\DS-projects\k-drama RS\data\koreanTV.csv")
#k_drama_df.head()


# In[63]:


#k_drama_df.info()


# In[64]:


#k_drama_df.isna().sum()


# In[65]:


k_df=k_drama_df
newShortStory = k_df['Short Story']
#newShortStory.head()


# In[66]:


ps = PorterStemmer()
def remove_splChar_normalizeWords(ss_line):
    word_list = []
    ss_line = re.sub('[^A-Za-z0-9]',' ', ss_line)
    #ss_line.replace('\n','')
    for word in ss_line.split():
        word_list.append(ps.stem(word))
        
    return  " ".join(word_list)


# In[67]:


newShortStory = k_df['Short Story']
newShortStory = newShortStory.apply(remove_splChar_normalizeWords)


# In[68]:


k_df['Modified Short Story'] = newShortStory


# In[113]:


#k_df.head()


# In[70]:


k_df=k_df.drop(['Votes:', 'Time','Short Story'],axis=1)
k_df.columns.values


# In[71]:


k_df['Tags'] =  [g.replace(',','') for g in k_df['Genre']]


# In[72]:


k_df['Tags'] = k_df['Tags']+" "+ [g.replace(',','') for g in k_df['Stars']] +" "+k_df['Modified Short Story']


# In[73]:


k_df['Tags'] = [word.lower() for word in k_df['Tags']]
k_df['Title_low']=[title.lower() for title in k_df['Title']]


# In[74]:


# creating vectorizer (with stopwords as well)
cv = CountVectorizer(max_features = 7000, stop_words = "english")
vect_mat = cv.fit_transform(k_df["Tags"]).toarray()


# In[75]:


cv.get_feature_names_out()


# In[76]:


similarity = cosine_similarity(vect_mat)


# In[135]:


#get input for recommondation
def genre_recomm(input_title):
    movieList=[]
    if(input_title.lower() not in k_df['Title_low'].values):
        return 'The k-drama you like is not in the data base try another name'
    
    series_index = k_df[k_df["Title_low"] == input_title.lower()].index[0]
    
    # Calculate similarity
    distances = similarity[series_index]
    series_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
    #print('You may also like:')
    #print(("Name"+'\t'+"Rating").expandtabs(30))
    # For all the similar series, print their name
    for i in series_list:      
        movieList.append(k_df.iloc[i[0]]["Title"])  
        #print((k_df.iloc[i[0]]["Title"] + '\t' + k_df.iloc[i[0]]["Rating"]).expandtabs(30))
        print(i)
        
    return movieList

def getAllTitlesAvailable():
    return k_df['Title'].values