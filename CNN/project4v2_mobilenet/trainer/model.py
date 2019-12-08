import numpy as np

from keras.models import *
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
import keras

from keras.models import model_from_json
import tensorflow as tf
import cv2
import pickle

#from joblib 
from tensorflow.python.lib.io import file_io
import argparse


def build():
  #load mobilenet architecture from keras
  #top false to get it to classify with our application
  base_model = keras.applications.mobilenet.MobileNet(include_top=False,input_shape=(224,224,3))
  model = base_model.output


  print("good loading of mobilent")

  #make it more fine grain analysis
  model = GlobalAveragePooling2D()(model)
  model = Dense(1024,activation='relu')(model)
  model = Dense(1024,activation="relu")(model)
  model = Dense(512,activation="relu")(model)


  predictions = Dense(10,activation='softmax')(model)
  model = Model(base_model.input, predictions)


  #do not train the first 20 layers of the model
  for layer in model.layers[:20]:
    layer.trainable=False
  for layer in model.layers[20:]:
    layer.trainable=True

  model.summary()

  print("good adding")
  

  model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])



  return model



def pickleLoader():
  #file batches
  feat_batches = ['gs://project4_datas/trainDataFeat_1/feat_data7594_15189', 'gs://project4_datas/trainDataFeat_1/feat_data15189_22784', 'gs://project4_datas/trainDataFeat_1/feat_data22784_30379', 'gs://project4_datas/trainDataFeat_1/feat_data30379_37974', 'gs://project4_datas/trainDataFeat_1/feat_data37974_45569', 'gs://project4_datas/trainDataFeat_1/feat_data45569_53164', 'gs://project4_datas/trainDataFeat_1/feat_data53164_60759', 'gs://project4_datas/trainDataFeat_1/feat_data60759_68354', 'gs://project4_datas/trainDataFeat_1/feat_data68354_75949', 'gs://project4_datas/trainDataFeat_1/feat_data75949_83544']
  label_batches = ['gs://project4_datas/trainDataLabel_1/label_data7594_15189', 'gs://project4_datas/trainDataLabel_1/label_data15189_22784', 'gs://project4_datas/trainDataLabel_1/label_data22784_30379', 'gs://project4_datas/trainDataLabel_1/label_data30379_37974', 'gs://project4_datas/trainDataLabel_1/label_data37974_45569', 'gs://project4_datas/trainDataLabel_1/label_data45569_53164', 'gs://project4_datas/trainDataLabel_1/label_data53164_60759', 'gs://project4_datas/trainDataLabel_1/label_data60759_68354', 'gs://project4_datas/trainDataLabel_1/label_data68354_75949', 'gs://project4_datas/trainDataLabel_1/label_data75949_83544']

  for i in range(len(feat_batches)):
    
    #load every pickle file name from the directory/bucket
    with file_io.FileIO(feat_batches[i],mode="r") as f:
      with file_io.FileIO(label_batches[i],mode="r") as l:
        yield pickle.load(f),pickle.load(l)




def main(train_file="training_data_all2.pickle" ,**args):
  num_classes=10
  tensorboard = keras.callbacks.TensorBoard(log_dir="gs://project4_datas/output/logs/", histogram_freq=0, write_graph=True, write_images=True)
  model = build()
  print("model returned")
  for X,y in pickleLoader():
    X = X/255.0
    y = np_utils.to_categorical(y,num_classes)
    model.fit(X,y, batch_size = 40 ,epochs=18,validation_split=0.20,shuffle=True,callbacks=[tensorboard])



  #saves weights and architecture
  model.save('model.h5')
  with file_io.FileIO('model.h5', mode='r') as input_f:
    with file_io.FileIO(job_dir + '/model-try.h5',mode='w+') as output_f:
      output_f.write(input_f.read())
  print("SAVED")
    


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--train-file',
    help='GCS location of training path',
    required=True
  )

  
  parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models',
    required=True
  )
  args = parser.parse_args()
  arguments = args.__dict__
  job_dir = arguments.pop('job_dir')

  main(**arguments)  