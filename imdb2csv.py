from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
import time
from datetime import datetime

import cv2
import dlib
import numpy as np
import pandas as pd
from imutils.face_utils import FaceAligner
from scipy.io import loadmat
from sklearn.model_selection import train_test_split




def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
    data = {"file_name": full_path, "gender": gender, "age": age, "score": face_score,
            "second_score": second_face_score}
    dataset = pd.DataFrame(data)
    #print(dataset)
    return dataset



def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1





def main(db_path, db_name, test_size, face_width, max_age, min_age, threadID):
    start_time = time.time()
    def imagePath2string(path):
        #print("fuck")
        shape_predictor = 'shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor)
        fa = FaceAligner(predictor, desiredFaceWidth=face_width)  
        
        image = cv2.imread("data/imdb_crop/"+path[0], cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 2)
        if len(rects) != 1: 
            print("NLLL")
            return "NULL"
        image_raw = fa.align(image, gray, rects[0])
        image_raw = image_raw.tostring()
        print(len(image_raw))
        return image_raw

    data_sets = get_meta(db_path, db_name)
    print(data_sets.shape)
    print("main")

    data_sets = data_sets[data_sets.score > 0.75]
    data_sets = data_sets[data_sets.age <= max_age]
    data_sets = data_sets[data_sets.age >= min_age]
    

    m,n=data_sets.shape
    print(data_sets.shape)
    data_sets['file_name']=data_sets['file_name'].apply(imagePath2string)
    data_sets = data_sets.drop(['gender','score','second_score'],axis=1)    
    data_sets = data_sets[data_sets.file_name != 'NULL']

    print(data_sets.shape)
    
    train_sets, test_sets = train_test_split(data_sets, test_size=test_size, random_state=2017)
    #data_sets.to_csv('dataset.csv',header=True,index=False, sep=",")
    train_sets['age'].to_csv("myData/imdbtrain_label"+str(threadID),header=True,index=False, sep=",")
    m,n=train_sets.shape
    F = open("myData/imdbtrain_set"+str(threadID),"w") 
    for i in range(m):
        F.write(train_sets['file_name'].values[i])
    F.close()

    test_sets['age'].to_csv('myData/imdbtest_label'+str(threadID),header=True,index=False, sep=",")
    m,n=test_sets.shape
    F = open("myData/imdbtest_set"+str(threadID),"w") 
    for i in range(m):
        F.write(test_sets['file_name'].values[i])
    F.close()

    print(train_sets.shape,test_sets.shape)
    duration = time.time() - start_time
    print("Running %.3f sec All done!" % duration)
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_db", type=str, default="./data/imdb_crop/imdb.mat")
    parser.add_argument("--test_size", type=float, default=0.01, help="How many items as testset")
    parser.add_argument("--face_width", type=int, default=160, help="dlib_detect_face_width")  
    parser.add_argument("--max_age", type=int, default=100, help="maxAgeInTheTrainData")
    parser.add_argument("--min_age", type=int, default=0, help="minAgeInTheTrainData")  
    parser.add_argument("--threadID", type=int, default=0, help="ID")
    args = parser.parse_args()


    print("Using imdb dataset")
    main(db_path=args.wiki_db, db_name="imdb", test_size=args.test_size, face_width=args.face_width, max_age=args.max_age, min_age=args.min_age, threadID=args.threadID)

