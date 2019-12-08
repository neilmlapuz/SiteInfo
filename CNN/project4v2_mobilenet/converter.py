from tensorflow.contrib import lite
import numpy as np
import keras.models
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf
import keras
from keras.applications import MobileNet
from keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import CustomObjectScope
import cv2

##########################################
#conver h5 model to tflite model with the# 
#use of tfliteconverter from tflite      #
##########################################

#load relu6 from tf keras
def relu6(x):
  return K.relu(x, max_value=6)
  
  
with CustomObjectScope({'relu6': relu6}):
  converter = lite.TFLiteConverter.from_keras_model_file( 'final_model.h5' )

  model = converter.convert()
  file = open( 'final.tflite' , 'wb' ) 
  file.write( model )
