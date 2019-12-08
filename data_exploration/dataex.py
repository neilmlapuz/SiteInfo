import pandas as pd
import matplotlib.pyplot as plt

def get_total(tf):
    
    print("{} {}".format("Train Data Shape Rows:", tf.shape[0]))
    print("{} {}".format("Train Data Shape Columns:", tf.shape[1]))


def get_head(tf):
    print(tf.head())

def get_nuniquev(tf):
    print(tf.nunique())

def get_frequent(tf):
    tf = tf[(tf.landmark_id != "None")]

    frequent10 = pd.DataFrame(tf.landmark_id.value_counts().head(10)).dropna()
    frequent10.reset_index(level=0, inplace=True)
    frequent10.columns = ['landmark_id','count']

    print(frequent10)
    return frequent10

def plot(df):
    df.groupby("landmark_id")["count"].plot(kind="bar")
    

def get_file():
    train_file  = pd.read_csv('train.csv')

    return train_file

def main():
    tf  = get_file()

    get_total(tf)
    print("------------------------")
    get_head(tf)
    print("------------------------")
    get_nuniquev(tf)
    print("------------------------")
    df = get_frequent(tf)
    plot(df)












if __name__ == "__main__":
    main()