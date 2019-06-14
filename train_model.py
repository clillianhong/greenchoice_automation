import fastai
from fastai import *
from fastai.text import * 
import pandas as pd
import numpy as np
from functools import partial
import io
import os
import nltk 
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# test set: https://forums.fast.ai/t/how-to-add-a-test-set/38956/7
# text.data reference methods: https://docs.fast.ai/text.html

'''
csv_path >> path to a CSV that has two columns ['text', 'labels'] where 'labels' are the classification categories and 'text' are the features in a string
save_path >> path/name to save the model to 
lr >> learning rate, can be found experimentally with learn.lr_find(start_lr=1e-5, end_lr=10000) and then learn.recorder.plot()
dropout >> percent dropout
test_size >> fraction size of validation set
batch >> batch size 
'''

def trainModel(csv_path, save_path, lr=1e-1, dropout=0.5, test_size=0.3, batch=32):

	data = pd.read_csv(csv_path)
	df = pd.DataFrame({'label':data.labels, 'text':data.text})

	#removing all special characters and spaces, replacing with a single space 
	df['text'] = df['text'].str.replace("[^a-zA-Z]", " ") 

	stop_words = stopwords.words('english') 

	#tokenization 
	tokenized_doc = df['text'].apply(lambda x: x.split())

	#remove stop-words
	tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

	#detokenize
	detokenized_doc = []
	for i in range(len(df)):
	    t = ' '.join(tokenized_doc[i])
	    detokenized_doc.append(t)
	    
	df['text'] = detokenized_doc

	# split data into training and validation set
	df_trn, df_val = train_test_split(df, stratify = df['label'], test_size = test_size, random_state = 12)
	df_trn.shape, df_val.shape

	#language model data - vectorization 
	data_lm = TextLMDataBunch.from_df(train_df=df_trn, valid_df=df_val, path = "")

	# Classifier model data - 
	data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=batch)

	#TRAIN LANGUAGE MODEL------------------------------------------------------------------
	learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=dropout)

	learn.fit_one_cycle(1, lr, moms=(0.8,0.7))
	learn.unfreeze()
	learn.fit_one_cycle(10, lr/10, moms=(0.8,0.7)) 

	learn.save_encoder('ft_enc_best')

	#TRAIN CLASSIFIER MODEL-----------------------------------------------------------------
	learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=dropout)
	learn.load_encoder('ft_enc_best')
	learn.freeze()


	learn.lr_find()
	learn.recorder.plot()

	learn1 = learn
	learn1.fit_one_cycle(1, lr/10, moms=(0.8,0.7))
	learn1.save('clas_first') 

	learn1.freeze_to(-2)
	learn1.fit_one_cycle(1, lr/10, moms=(0.8,0.7))  #decrease learning rate by a factor of 10? 
	learn1.save('clas_first_1')

	learn1.freeze_to(-3)
	learn1.fit_one_cycle(1, lr/100, moms=(0.8,0.7))  #decrease learning rate to middle of spike down? (factor of 10?)
	learn1.save('clas_first_2')

	learn1.unfreeze()
	learn1.fit_one_cycle(1, lr/100, moms=(0.8,0.7))  #decrease lr slightly further


	learn1.save(save_path)
	print("Model successfully saved to: " + save_path)


	interp = ClassificationInterpretation.from_learner(learn1)
	interp.plot_confusion_matrix()

	print("Confusion matrix plotted.")


def main():
	csv_path = "~/Experiment/experiment/sample_prod_cat_ingredients.csv"
	save_path = "testing_model"
	trainModel(csv_path, save_path)

if __name__ == '__main__':
	main()