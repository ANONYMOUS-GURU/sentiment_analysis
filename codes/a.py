import tensorflow as tf 
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Conv1D,Bidirectional,Flatten
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import os
import numpy as np
from data_helper import generate_train_val,yielder,text_tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard


def step_decay(curr_epoch,curr_lr):	
	return curr_lr/2

def load_pretrained_embedding(PATH,tokenizer,output_dim):
	embeddings_index=dict()
	with open(PATH) as f:
		for line in f:
			values=line.split()
			word=values[0]
			coefs=np.asarray(values[1:],dtype=np.float32)
			embeddings_index[word]=coefs

	print('LOADED WORD VECTORS :: Size --> {} '.format(len(embeddings_index)))

	vocab_size=len(tokenizer.word_index)+1

	embedding_matrix = np.zeros((vocab_size, output_dim))
	for word, i in tokenizer.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	return embedding_matrix



train,val,maxlen=generate_train_val(PATH=os.path.join(os.getcwd(),'sentiment140','data.csv'),val_frac=0.1)
maxlen=75
# tokenizer=text_tokenizer(train)

# with open('tokenizer.pkl','wb+') as f:
# 	pickle.dump(tokenizer,f)

with open('tokenizer.pkl','rb') as f:
	tokenizer=pickle.load(f)

train_data_yielder=yielder(train,tokenizer=tokenizer,batch_size=32,max_length=maxlen)
val_data_yielder=yielder(val,tokenizer=tokenizer,batch_size=32,max_length=maxlen)
embedding_matrix=load_pretrained_embedding(os.path.join('glove.6B.50d.txt'),tokenizer,50)
vocab_size=len(tokenizer.word_index)+1
batch_size_val=64
batch_size_train=32
num_batches_train=(int)(np.ceil(train.shape[0]/batch_size_train))
num_batches_val=(int)(np.ceil(val.shape[0]/batch_size_val))


def build_model(maxlen,vocab_size,embedding_matrix):
	visible=Input(shape=(maxlen,))
	embedding=Embedding(input_dim=vocab_size,output_dim=50,weights=[embedding_matrix],
		input_length=maxlen,trainable=False)(visible)
	conv1=Conv1D(64,5,activation='relu',padding='VALID')(embedding)
	lstm1=Bidirectional(LSTM(32,return_sequences=True,activation='tanh'))(conv1)
	flatten=Flatten()(lstm1)
	outputs=Dense(2)(flatten)	

	model=Model(inputs=visible,outputs=outputs)
	return model

model=build_model(maxlen,vocab_size,embedding_matrix)

checkpoint_addr=os.path.join('checkpoints','weightsbest.hdf5')
if os.path.exists(checkpoint_addr):
	model.load_weights(checkpoint_addr,by_name=True)
	print('LOADED MODEL WEIGHTS')
checkpoint = ModelCheckpoint(checkpoint_addr, monitor='val_accuracy', verbose=1, 
	save_best_only=True, mode='max')
earlystopping=EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto', baseline=None, 
	restore_best_weights=True)
lrschedule=LearningRateScheduler(step_decay, verbose=1)
csvlog=CSVLogger('training_log.csv', append=False)
tensorboard=TensorBoard(log_dir='./logs',batch_size=32, write_graph=True,update_freq=100)

callbacks_list = [checkpoint,earlystopping,lrschedule,csvlog,tensorboard]

print(model.summary())
plot_model(model,to_file='a.png')

if not os.path.exists(os.path.join('checkpoints')):
	os.mkdir(os.path.join('checkpoints'))


optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
loss=SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

# for x in range(num_batches_train):
# 	train_x,train_y,weights=next(train_data_yielder)
# 	history=model.train_on_batch(train_x,train_y,reset_metrics=False)
# 	print(history)

history=model.fit(train_data_yielder,epochs=3,steps_per_epoch=num_batches_train,validation_data=val_data_yielder,
	validation_steps=num_batches_val,verbose=1,callbacks=callbacks_list)





