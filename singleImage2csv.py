#encoding=utf-8
'''
将单张图片读入并处理成想要的格式，以便于测试
'''

import cv2
import dlib
import numpy as np
import pandas as pd
from imutils.face_utils import FaceAligner
import argparse
import time



def main(imagePath,faceWidth,savePath):
    #print("fuck")
    start_time = time.time()
    shape_predictor = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    fa = FaceAligner(predictor, desiredFaceWidth=faceWidth)  
    
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    if len(rects) != 1: 
        print("NLLL")
        return "NULL"
    image_raw = fa.align(image, gray, rects[0])
    image_raw = image_raw.tostring()
    imageLen=len(image_raw)
    print(imageLen)
    imageList=[]
    for i in range(imageLen):
        imageList.append(ord(image_raw[i]))
    myImage=np.array(imageList).reshape(1,faceWidth*faceWidth*3)
    myImage.astype(int)
    
    np.savetxt(savePath, myImage, fmt="%d", delimiter=",")
    print("finish save",savePath)
    duration = time.time() - start_time
    print("Running %.3f sec All done!" % duration)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, default="parker.jpg")
    parser.add_argument("--faceWidth", type=int, default=70, help="dlib_detect_face_width") 
    parser.add_argument("--savePath", type=str, default="curImage.csv") 
   
    args = parser.parse_args()

    main(imagePath=args.imagePath, faceWidth=args.faceWidth, savePath=args.savePath)
