
import numpy as np 
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Conv1D,Bidirectional,Flatten
from tensorflow.keras import Model


def build_model(build_keys):
	print(build_keys)
	embedding_matrix,vocab_size=load_pretrained_embedding(PATH=build_keys['path_embedding'],tokenizer=build_keys['tokenizer'],output_dim=build_keys['output_dim'])
	visible=Input(shape=(build_keys['maxlen'],))
	embedding=Embedding(input_dim=vocab_size,output_dim=50,weights=[embedding_matrix],
		input_length=build_keys['maxlen'],trainable=False)(visible)
	conv1=Conv1D(64,5,activation='relu',padding='VALID')(embedding)
	lstm1=Bidirectional(LSTM(32,return_sequences=True,activation='tanh'))(conv1)
	flatten=Flatten()(lstm1)
	outputs=Dense(2)(flatten)	

	model=Model(inputs=visible,outputs=outputs)
	return model

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

	return embedding_matrix,vocab_size