from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os

def softmax(list_):
    return np.asarray([np.exp(x)/np.sum(np.exp(x), axis=0) for x in list_])

class test_model:
	def __init__(self,model_path,dict_class,utils_path):
		self.model=load_model(model_path)
		with open(utils_path,'rb') as f:
			self.model_keys=pickle.load(f)
		self.dict_class=dict_class

	def predict(self,X):
		X=np.asarray(X)
		#Processed Input
		X=pad_sequences(self.model_keys['tokenizer'].texts_to_sequences(X),maxlen=self.model_keys['maxlen'],padding='post')
		logits=self.model.predict(X)
		probs=softmax(logits)
		out_class=[self.dict_class[x] for x in np.argmax(probs,axis=1)]

		print(X)
		print(logits)
		print(out_class)


if __name__=='__main__':

	dict_class={0:'negative',1:'positive'}
	model_path=os.path.join('..','output_data','h5_file','model.h5')
	utils_path=os.path.join('..','output_data','train_keys.pkl')
	model=test_model(model_path=model_path,dict_class=dict_class,utils_path=utils_path)
	
	X=['I am horny']
	model.predict(X)