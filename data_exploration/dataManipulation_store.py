import os
import numpy as np 
import cv2

import random
import pickle

#preprocess image data to appropriate data - np array to feed to NN and save it to pickle file to access when training


def getTrainingData():
    train = []
    dict_files = {}
    traindir = "/mnt/project4/data_all"
    #classes = ["HagiaSophia","Colosseum"]
    classes = ["Colosseum","HagiaSophia","MNDC","Pantheon","Petronas","RialtoB","StPeterB","charlesB","hofP","alhambra"]
    for categ in classes:
        path = os.path.join(traindir,categ)
        class_label = classes.index(categ)
        for img in os.listdir(path):
            train.append([str(os.path.join(path,img)), class_label])
            

    random.shuffle(train)
    del train[:41]
    #print(train)
    print(len(train))
    
    #img_arr = cv2.imread(train[2][0],cv2.IMREAD_COLOR)
    #print(img_arr)


    step = 7595
    #goes in steps of 7595 - for each pickle file saved, there will be at least 7500 images
    #strore and access in batches 
    for j in range(0, len(train),step):
        print(j)
        feats = []
        labels= []
        for i in range(j, j+step):
            try:
            #reading with cv and resizing to appropriate dimension for the model
                img_arr = cv2.imread(train[i][0],cv2.IMREAD_COLOR)
                new_arr = cv2.resize(img_arr,(224,224))
                feats.append(new_arr)
                labels.append(train[i][1])
            except Exception as e:
                print("Errpr", train[i][0])
                pass
        #write pickle files
        with open('./trainDataFeat_1/feat_data'+str(i)+"_"+str(i+step),"wb") as f:
            pickle.dump(np.array(feats).reshape(-1,224,224,3),f,protocol=2)
        with open('./trainDataLabel_1/label_data'+str(i)+"_"+str(i+step),"wb") as l:
            pickle.dump(labels, l, protocol=2)



def main():
    getTrainingData()

if __name__ == "__main__":
    main()
