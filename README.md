# Sentiment_analysis

The above model works on a sentiment analysis on 1.4M tweets into 2 classes namely Posititve and Negative.




The embedding can also be downloaded from the same link.

## Dependencies:
tensorflow(2.1)

## Description:

Download the data and embedding and put them in the manner shown below.
#### To train the above model run the train.py file inside the codes folder.
You can download the training dataset from the google drive link given here.
https://drive.google.com/drive/folders/1JbaMolCoqdWw3hI2Yy0Keu5MLjsK6jyv?usp=sharing

Arrange your directory in the given manner.
https://github.com/ANONYMOUS-GURU/sentiment_analysis/blob/master/dir_str.png

### Outputs
Next run train.py to train the model which will generate the output_data folder with the tensorboard_logs output_logs and also the pickle tokenizer for inference.The saved_model is also saved in the folder.


## Architecture
The model architecture can be seen below:
![](https://github.com/ANONYMOUS-GURU/sentiment_analysis/blob/master/codes/model.png)

## Prediction:
To predict from a saved model use the predict.py function and to generate the predictions.

## Current State:
The model achieves around 80% accuracy with the given architecture and embedding. Increasing the size of the embedding may help generate better results. More embedding files can be dowloaded from the link https://nlp.stanford.edu/projects/glove/

## Future Goals:
You can change the embedding file or train your own embedding.That is something to be experimented with.



