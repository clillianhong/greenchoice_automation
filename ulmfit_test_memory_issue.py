#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import fastai
from fastai import *
from fastai.text import * 
import pandas as pd
import numpy as np
from functools import partial
import io
import os


# PREPROCESSING DATA 

# In[2]:


from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data


# In[3]:


df = pd.DataFrame({'label':dataset.target, 'text':dataset.data})


# In[4]:


df.shape


# In[5]:


df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")


# In[6]:


import nltk 

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#tokenization 
tokenized_doc = df['text'].apply(lambda x: x.split())

#remove stop-words
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])


# In[7]:


#de-tokenization

detokenized_doc = []
for i in range(len(df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
    
df['text'] = detokenized_doc


# In[8]:


from sklearn.model_selection import train_test_split

# split data into training and validation set
df_trn, df_val = train_test_split(df, stratify = df['label'], test_size = 0.4, random_state = 12)


# In[9]:


df_trn.shape, df_val.shape


# In[10]:


#language model data
data_lm = TextLMDataBunch.from_df(train_df=df_trn, valid_df=df_val, path = "")

# Classifier model data
data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)


# TRAINING LANGUAGE MODEL 

# In[ ]:


# learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.3)
# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


# # train the learner object with learning rate = 1e-2
# learn.fit_one_cycle(1, 5e-2, moms=(0.8,0.7))


# In[ ]:


# learn.unfreeze()
# learn.fit_one_cycle(10, 1e-2, moms=(0.8,0.7)) #decrease learning rate by a factor of 10? why? 


# In[ ]:


# learn.save_encoder('ft_enc')


# TRAINING CLASSIFIER

# In[11]:


learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')
# learn.freeze()


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# 

# learn.fit_one_cycle(1, 5e-2, moms=(0.8,0.7))  
# learn.save('clas_first')
# learn.load('clas_first')

# In[ ]:


# learn.fit_one_cycle(1, 5e-2, moms=(0.8,0.7))
# learn.save('clas_first') 


# In[ ]:


# learn.load('clas_first')
# learn.freeze_to(-2)
# learn.fit_one_cycle(1, 5e-3, moms=(0.8,0.7))  #decrease learning rate by a factor of 10? 
# learn.save('clas_first_1')


# In[ ]:


# learn.load('clas_first_2')


# In[ ]:


# learn.freeze_to(-3)
# learn.fit_one_cycle(1, 5e-3, moms=(0.8,0.7))  #decrease learning rate to middle of spike down? (factor of 10?)
# learn.save('clas_first_3')
#MEMORY ALLOCATION ERROR https://github.com/fastai/fastai/issues/1979


# In[12]:


learn.load('clas_first_3')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(1, 5e-3, moms=(0.8,0.7))  #decrease lr slightly further


# In[ ]:




