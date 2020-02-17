import tensorflow as tf 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler,CSVLogger,TensorBoard,ModelCheckpoint

import pickle
import os
import numpy as np
from data_helper import helper
from model_utils import build_model

class make_model:

	def __init__(self,checkpoint_dir,checkpoint_addr,log_csv_addr,tensorboard_logs,
		train_keys_path,path_to_data,val_frac,final_model_path,restore):

		data_loader=helper(path_to_data,val_frac)
		self.train_generator,self.val_generator=data_loader.generator_train_val(train_keys_path=train_keys_path,restore=restore)
		build_keys={'tokenizer':data_loader.train_keys['tokenizer'],'path_embedding':os.path.join('..','embedding','glove.6B.50d.txt'),
		'output_dim':50,'maxlen':data_loader.train_keys['maxlen']}
		self.model=build_model(build_keys)
		self.num_batches_train=data_loader.num_batches_train
		self.num_batches_val=data_loader.num_batches_val

		self.checkpoint_dir=checkpoint_dir
		self.checkpoint_addr=checkpoint_addr
		self.log_csv_addr=log_csv_addr
		self.tensorboard_logs=tensorboard_logs
		self.final_model_path=final_model_path


	def lr_schedule(self,curr_epoch,curr_lr):
		if (curr_epoch+1)%5==0:	
			return curr_lr/2
		else:
			return curr_lr

	def get_callbacks(self):
		if not os.path.exists(self.checkpoint_dir):
			os.mkdir(self.checkpoint_dir)

		checkpoint = ModelCheckpoint(self.checkpoint_addr, monitor='val_accuracy', verbose=1, 
			save_best_only=True, mode='max')
		earlystopping=EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto', baseline=None, 
			restore_best_weights=True)
		lrschedule=LearningRateScheduler(self.lr_schedule, verbose=1)
		csvlog=CSVLogger(self.log_csv_addr, append=False)
		tensorboard=TensorBoard(log_dir=self.tensorboard_logs,batch_size=32, write_graph=True,update_freq=100)

		self.callbacks_list = [checkpoint,earlystopping,lrschedule,csvlog,tensorboard]

	def summarize(self):
		print(self.model.summary())
		plot_model(self.model,to_file='model.png')

	def compile_model(self):
		optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
		loss=SparseCategoricalCrossentropy(from_logits=True)
		self.model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

	def train_model(self):
		self.get_callbacks()
		self.compile_model()
		self.summarize()

		if os.path.exists(checkpoint_addr):
			self.model.load_weights(checkpoint_addr,by_name=True)
			print('LOADED MODEL WEIGHTS')
		else:
			print('TRAINING FROM SCRATCH')

		self.history=self.model.fit(self.train_generator,epochs=3,steps_per_epoch=self.num_batches_train,validation_data=self.val_generator,
			validation_steps=self.num_batches_val,verbose=1,callbacks=self.callbacks_list)

		self.export_model()

	def export_model(self):
		self.model.save(self.final_model_path)


# for x in range(num_batches_train):
# 	train_x,train_y,weights=next(train_data_yielder)
# 	history=model.train_on_batch(train_x,train_y,reset_metrics=False)
# 	print(history)


if __name__=='__main__':

	output_data_path=os.path.join('..','output_data')
	if not os.path.exists(output_data_path):
		os.mkdir(output_data_path)

	train_keys_path=os.path.join(output_data_path,'train_keys.pkl')
	checkpoint_dir=os.path.join(output_data_path,'checkpoints')
	checkpoint_addr=os.path.join(checkpoint_dir,'weightsbest.hdf5')
	log_csv_addr=os.path.join(output_data_path,'training_log.csv')
	tensorboard_logs=os.path.join(output_data_path,'tboard_logs')
	path_to_data=os.path.join('..','data','data.csv')
	final_model_path=os.path.join(output_data_path,'h5_file','model.h5')
	if not os.path.exists(os.path.join(output_data_path,'h5_file')):
		os.mkdir(os.path.join(output_data_path,'h5_file'))

	val_frac=0.1

	restore=False
	if os.path.exists(checkpoint_addr):
		restore=True

	mymodel=make_model(checkpoint_dir=checkpoint_dir,checkpoint_addr=checkpoint_addr,log_csv_addr=log_csv_addr,
		tensorboard_logs=tensorboard_logs,train_keys_path=train_keys_path,path_to_data=path_to_data,val_frac=val_frac,final_model_path=final_model_path,restore=restore)
	mymodel.train_model()


