import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

#use to manipulate the images that will be read and loaded
import cv2
import pickle

from tensorflow.python.lib.io import file_io
import argparse




def build(X,y):
    num_classes=2
    y = np_utils.to_categorical(y,num_classes)

    #add each layer in sequence and automatically get connected
    model = Sequential()

    model.add(Conv2D(32,(3,3),padding="same",input_shape= X.shape[1:],activation="relu"))
    model.add(Conv2D(32,(3,3),activation="relu"))
    #retrieve highest pixel value
    model.add(MaxPooling2D(pool_size=(2,2)))
    #regularisation
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), padding="same",activation="relu"))
    model.add(Conv2D(64,(3,3),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(512,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation="softmax"))

    #classify between two classes
    model.compile(loss="binary_crossentropy", optimizer="adam",metric=["accuracy"])
    model.summary()

    model.fit(X,y,batch_size=32, epochs=20,validation_data=0.20)



    return model

def splitData(training_data):
  img_size = 224
  X = []
  y = []
  for feat, label in training_data:
    X.append(feat)
    y.append(label)

  X = np.array(X).reshape(-1,img_size,img_size,1)

  return X,y

def main(train_file="training_data2.pickle",**args):

    #load training file
    input_file = file_io.FileIO(train_file,mode="r")
    input_file = pickle.load(input_file)
    #split training data and preprocess
    X,y = splitData(input_file)

    #make pixel values between 0 and 1
    X = X/255.0

    print("success")
    model = build(X,y)

    model.save('model.h5')
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5',mode='w+') as output_f:
            output_f.write(input_f.read())
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--train-file',
      help='GCS location of training path',
      required=True
    )

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments.pop('job_dir')

    main(**arguments)   

