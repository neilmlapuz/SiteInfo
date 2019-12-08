#Extracts a specific landmark data in the whole dataset
# and downloads only that data.

import pandas as pd
import numpy as np
import os, sys, requests, shutil
from urllib import request, error
from  collections import Counter
import multiprocessing


i = 0

def get_landmarkId(df):
    #Create a dataframe for a specific landmark id
    df = df[(df.landmark_id == "5554")]
    df = df.head(8350)
    #Gets how many rows
    print(df.shape)
    return df

#function to put specific landmark to csv file
def landmark_csv():
    df = pd.read_csv("train.csv")
    print(df.shape)
    #Dataframe that only consist of the target landmark
    landmark_frame = get_landmarkId(df)
    #Write to csv file without the index
    landmark_frame.to_csv("try.csv",index=False)


def get_image(url):
    #goes to server to request some resource
    res = requests.get(url,stream = True)
    global i 
    if os.path.exists("./Petronas/Petronas_"+str(i)+".jpg"):
        print(i, " already exist")
        i+=1
        return
    
    with open("./Petronas/Petronas_"+str(i)+".jpg", "wb") as out_file:
        shutil.copyfileobj(res.raw, out_file)
    print(i)
    i+=1

    del res


def main():
    #landmark_csv()
    data_spec_landmark = pd.read_csv("try.csv")
    urls = data_spec_landmark["url"]
    url_lst = [item for item in urls]
    pool = multiprocessing.Pool(processes=1)
    pool.map(get_image,url_lst)

    #print(type(urls))
    #Prints list of files in the current directory
    #print(os.listdir("./"))

    #iterate_links(urls)




if __name__ == "__main__":
    main()