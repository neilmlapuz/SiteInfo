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

from keras.preprocessing import image
from IPython.display import Image

from collections import defaultdict
from glob import glob

import cv2
import time

############################################################
# This code is to test input images on a trained           #
# neural network based on the architecture on project4v2   #
# which is mobilenet                                       #
############################################################


#load from keras preprocessing module to feed the image
def prepare_image(file):
    
    img = image.load_img("./"+file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)



def main():

    
    curr_model = "./models/final_model1.h5"
    start = time.time()
    #retrieving hyper parameters for mobilenet model
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        loaded_model = load_model(curr_model)

    #load weights
    loaded_model.load_weights(curr_model)

    #compile model
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("loaded good")
    

    d = defaultdict(list)
    files = ["col","HS","MNDC","Pan","Pet","rial","spb","cb","hofp","al"]
    classes = ["Colosseum","HagiaSophia","MNDC","Pantheon","Petronas","RialtoB","StPeterB","charlesB","hofP","alhambra"]
        

    
    i = 0
    for item in files:
        d[i] = glob(item+"*")
        i += 1

    total = 0
    for item in d:
        total += len(d[item])

    print(total)
    
    correct = 0
    for key in d:
        incorrect = 0
        print("*****************************")
        print("class",key)
        print("*****************************")
        for item in d[key]:
            #preprocess image
            preprocess_img = prepare_image(item)
            #predict the image
            predictions = loaded_model.predict(preprocess_img)
            #convert vector to list
            list_pred = list(predictions[0])
            maxi = max(list(predictions[0]))

            #match the highest index in the list of predictions with the correspoding key
            if key == list_pred.index(maxi):
                correct += 1

            else:
                incorrect +=1
                print(item)
        print("{} = {}".format("Total incorrect "+ classes[key],incorrect))

    print("{} out of {}".format(correct,total))
    """
    ---previous iteration---
    validation_path = "./validation"
    for img in os.listdir(validation_path):
        
        preprocess_img = prepare_image(os.path.join(validation_path,img))
        predictions = loaded_model.predict(preprocess_img)
        list_pred = list(predictions[0])
        maxi = max(list(predictions[0]))
        
        print("file:{} -- class {}  with {} accuracy".format(img,classes[list_pred.index(maxi)],maxi))
    """
    end = time.time()
    print(end-start)

if __name__ == "__main__":
    main()