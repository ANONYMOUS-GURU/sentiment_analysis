import pandas as pd 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

class helper:
	def __init__(self,PATH,val_frac=0.1):
		
		data=pd.read_csv(PATH)
		data=data.sample(frac=1.0)
		val_size=int(data.shape[0]*val_frac)

		self.val=data.iloc[:val_size,:]
		self.train=data.iloc[val_size:,:]
		self.batch_size_train=32
		self.batch_size_val=32

		self.num_batches_train=(int)(np.ceil(self.train.shape[0]/self.batch_size_train))
		self.num_batches_val=(int)(np.ceil(self.val.shape[0]/self.batch_size_val))

		self.train_keys={'tokenizer':None,'maxlen':None}

	def yielder(self,data,batch_size):
		tokenizer=self.train_keys['tokenizer']
		maxlen=self.train_keys['maxlen']
		i=0
		d_={0:0,4:1}
		while True:
			start=i*batch_size
			end=min((i+1)*batch_size,data.shape[0])
			return_batch=data.iloc[start:end,:]
			i+=1
			yield pad_sequences(tokenizer.texts_to_sequences(return_batch['text'].values),
				maxlen=maxlen,padding='post'),np.asarray([d_[x] for x in return_batch['target']]),[None]
			if end==data.shape[0]:
				i=0     

	def generator_train_val(self,train_keys_path,restore=False):
		if not restore:
			self.get_train_keys()
			self.save_train_keys(train_keys_path)
		else:
			self.restore_train_keys(train_keys_path)

		return self.yielder(data=self.train,batch_size=self.batch_size_train),self.yielder(data=self.val,batch_size=self.batch_size_val)

	# def generator_all_data(self):
	# 	self.get_train_keys()
	# 	return yielder(self.data)

	# FUNCTION TO GET ALL HELPER VALUES
	def get_train_keys(self):
		self.text_tokenizer(self.train)
		self.get_maxlen()

	# HELPERS
	def text_tokenizer(self,train):
		text=train['text'].values
		self.train_keys['tokenizer']=Tokenizer()        
		self.train_keys['tokenizer'].fit_on_texts(text)

	def get_maxlen(self):
		self.train_keys['maxlen']=(int)(self.train.text.str.len().median())

	def save_train_keys(self,path):
		with open(path,'wb+') as f:
			pickle.dump(self.train_keys,f)

	def restore_train_keys(self,path):
		with open(path,'rb') as f:
			self.train_keys=pickle.load(f)

		print('restored train keys')

