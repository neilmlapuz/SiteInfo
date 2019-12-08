import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras.models
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf
import keras
from keras.applications import MobileNet
from keras.applications import imagenet_utils
from keras.utils.generic_utils import CustomObjectScope

import cv2
import time

############################################################
# This code is to test input images on a trained           #
# neural network based on the architecture on project4     #
#                                                          #
############################################################
def testModel(loaded_model, imageFile):

    img_pred = cv2.imread(imageFile,0)

    classes = ["Colosseum","HagiaSophia"]
    #forces the image to have the input dimensions equal to those used in 
  
    if img_pred.shape != [224,224]:
        img2 = cv2.resize(img_pred,(224,224))
        img_pred = img2.reshape(-1,224,224)
    else:
        img_pred = img_pred.reshape(-1,224,224)  
    

    #Reshapes the data to a 4d tensor to feed our model
    img_pred = img_pred.reshape(1,224,224,1)
    
    pred = loaded_model.predict_classes(img_pred)

    #Determine the probability of it
    pred_proba = loaded_model.predict_proba(img_pred)
    pred_proba = "%.2f%%" % (pred_proba[0][pred] * 100)

    print(  "%s with probability of %s" % (classes[pred[0]],pred_proba))



def main():


    start = time.time()

    loaded_model = load_model("output_model.h5")
    #load weights into model
    loaded_model.load_weights("output_model.h5")

    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("loaded good")
    testModel(loaded_model, "HS_TEST_PIC11.jpg")
    end = time.time()
    print(end-start)

if __name__ == "__main__":
    main()