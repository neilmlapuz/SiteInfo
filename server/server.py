from flask import Flask, request,jsonify,Response,redirect,url_for,send_file
import numpy as np
import cv2
import tensorflow as tf
import keras
import keras.models
from keras.models import load_model
import os
from keras import backend as K
from keras.applications import MobileNet
from keras.applications import imagenet_utils
from keras.utils.generic_utils import CustomObjectScope

from keras.preprocessing import image
from IPython.display import Image

import requests

import re



app = Flask(__name__)
@app.route("/")
def hello():
    return "Hello Neil"

@app.route("/search",methods=["GET","POST"])
def searchInfo():
    d = {"Colosseum":"Colosseum Rome","Hagia Sophia":"Hagia Sophia",
            "Museu Nacional d'Art de Catalunya":"Museu Nacional de Catalunya",
        "Pantheon":"Pantheon Rome","Petronas Twin Towers":"Petronas Twin Towers","The Rialto Bridge":"rialto bridge venice"
        ,"St. Peter's Basilica":"st peter basilica","Charles Bridge":"charles bridge","The Hofburg Palace":"hofburg palace"
        ,"The Alhambra":"the alhambra"}

    if request.method == "POST":
        search_term = request.json["result"]
        result = requests.get("https://www.google.com/search?q="+d[search_term]).text
        r= re.findall(r'\<div(.*?)id=\"rhs_block\"(.*?)script',result)
        #r = re.findall(r'\<ol\>(.*?)',result)
        r = r[0][1]
        s = re.findall(r'\<div(.*?)class=\"V7Q8V\"(.*?)',r)
    
        feat = re.findall(r'\<span\>(.*?)\<\/span\>',s[3][0])[0]
        rating = re.findall(r'\<span(.*?)class=(.*?)\>(.*?)\<\/span\>',s[2][0])[0][2]
        info = re.findall(r'\<span(.*?)class=(.*?)\>(.*?)\<\/span\>',s[5][0])[0][0]
        neg = re.compile(r'\<a|\>')
        info = re.sub(neg,"",info)
        addr = re.findall(r'\<span(.*?)class=(.*?)\>(.*?)\<\/span\>',s[7][0])[1][2]

    return jsonify(name=search_term,feat=feat,rating=rating,info=info,addr=addr)




#Dependability Engineering purpose
#If the classifier integrated on the device fail for whatever reason - this will be used as backup
@app.route("/test",methods=["GET","POST"])
def testit():
    img = "test.jpg"
    if request.method == "POST":
        file=request.files["fileUpload"]
        filename = file.filename
        file.save(os.path.join("./",img))
    
    return predict()
    #return send_file(img, mimetype="image/jpg")

def loadModel():
    model = "final_model1.h5"
    with CustomObjectScope({'relu6': keras.layers.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
        loaded_model = load_model(model)
    loaded_model.load_weights(model)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model

#change image to the appropriate representation to feed modeol
def preProcessImg():
    file="test.jpg"
    img = image.load_img("./"+file, target_size=(224, 224))
    img = img.rotate(270)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

#get highest probability value
def getHighestIndex(pred):
    highest = max(pred)
    return pred.index(highest),highest


@app.route("/submit",methods=["GET","POST"])
def predict():
    
    img = "test.jpg"
    if request.method == "POST":
        file=request.files["fileUpload"]
        filename = file.filename
        file.save(os.path.join("./",img))
    
    loaded_model = None
    classes = ["Colosseum","HagiaSophia","MNDC","Pantheon","Petronas","RialtoB","StPeterB","charlesB","hofP","alhambra"]
    outputImg = preProcessImg()

    loaded_model = loadModel()
    #Determine the probability of it
    pred = loaded_model.predict(outputImg)
    result,prob = getHighestIndex(list(pred[0]))
    prob = str(round(prob,2))
    K.clear_session()
    #return str(pred[0])
    return classes[result]
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=80)


