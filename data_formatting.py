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


# test set: https://forums.fast.ai/t/how-to-add-a-test-set/38956/7
# text.data reference methods: https://docs.fast.ai/text.html

# In[2]:


data = pd.read_csv("sample_prod_cat_ingredients.csv")


# In[3]:


df = pd.DataFrame({'label':data.labels, 'text':data.text})


# In[4]:


df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")
import nltk 

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#tokenization 
tokenized_doc = df['text'].apply(lambda x: x.split())

#remove stop-words
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])


# In[5]:



detokenized_doc = []
for i in range(len(df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
    
df['text'] = detokenized_doc


# In[6]:


from sklearn.model_selection import train_test_split

# split data into training and validation set
df_trn, df_val = train_test_split(df, stratify = df['label'], test_size = 0.3, random_state = 12)
df_trn.shape, df_val.shape


# In[7]:


#language model data - vectorization 
data_lm = TextLMDataBunch.from_df(train_df=df_trn, valid_df=df_val, path = "")

# Classifier model data - 
data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)


# In[8]:


learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.5)


# In[9]:


learn.lr_find(start_lr=1e-5, end_lr=10000)
learn.recorder.plot()


# In[10]:


learn.fit_one_cycle(1, 1e-1, moms=(0.8,0.7))


# In[11]:


learn.unfreeze()
learn.fit_one_cycle(10, 1e-2, moms=(0.8,0.7)) #decrease learning rate by a factor of 10? why? 


# In[12]:


learn.save_encoder('ft_enc_best')


# In[ ]:





# In[13]:


learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc_best')
learn.freeze()


# In[14]:


learn.lr_find()
learn.recorder.plot()


# In[15]:


learn1 = learn
learn1.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learn1.save('clas_first') 


# In[16]:


learn1.freeze_to(-2)
learn1.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))  #decrease learning rate by a factor of 10? 
learn1.save('clas_first_1')


# In[17]:


learn1.freeze_to(-3)
learn1.fit_one_cycle(1, 5e-3, moms=(0.8,0.7))  #decrease learning rate to middle of spike down? (factor of 10?)
learn1.save('clas_first_2')


# In[18]:


learn1.unfreeze()
learn1.fit_one_cycle(1, 5e-3, moms=(0.8,0.7))  #decrease lr slightly further


# In[ ]:





# In[ ]:





# In[20]:





# In[21]:


learn1.save("pantry_with_ingred_model_1")


# In[22]:


interp = ClassificationInterpretation.from_learner(learn1)
interp.plot_confusion_matrix()


# In[23]:


learn1.fit_one_cycle(1, 5e-3, moms=(0.8,0.7)) 


# In[24]:


interp = ClassificationInterpretation.from_learner(learn1)
interp.plot_confusion_matrix()


# In[45]:





# In[26]:


learn1.predict("Planters Salted Caramel Nuts - 5oz")


# In[27]:


learn1.predict("Planters NUT-rition Essential Nutrients Deluxe Nut Mix - 5.5oz")


# In[28]:


learn1.predict("Gummi Worms - 7oz - Market Pantry™")


# In[29]:


learn1.predict("HERSHEY'S Miniatures Party Bag Assorted Chocolate Candy Bars - 40oz")


# In[30]:


learn1.predict("Dum Dums Original Assorted Flavors Lollipops - 1000ct")


# In[31]:


learn1.predict("JOLLY RANCHER Original Flavors Hard Candies - 3.75lbs")


# In[32]:


learn1.predict("Almonds, Peanuts & Sea Salt with Cocoa Drizzle Nut Bars - 12ct - Simply Balanced™")


# In[33]:


learn1.predict("Kashi Dark Mocha Almond Chewy Granola Bars - 6ct")


# In[34]:


learn1.predict("Organic Roasted Restaurant Style Salsa 24oz - Simply Balanced™")


# In[35]:


learn1.predict("Pace® Mild Chunky Salsa 64 oz")


# In[36]:


learn1.predict("Pepperidge Farm® Pirouette® Crème Filled Wafers Chocolate Hazelnut Cookies, 13.5oz Tin")


# In[37]:


learn1.predict("Milkmakers Oatmeal Chocolate Chip Cookies - 10ct")


# In[38]:


learn1.predict("Skinny Pop Popcorn 100 Calorie Bags - 24ct")


# Keep in mind:
#     - preprocess the input data the exact same way
#     - might want to hardcode, create list of edge case brands/items that can't be predicted well
#     - might want to generate data for edge case subtypes
#     - try weighting in the future (concatenate data within python to manage this as a hyperparameter)
