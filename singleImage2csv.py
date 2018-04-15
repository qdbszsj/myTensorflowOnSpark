#encoding=utf-8
'''
将单张图片读入并处理成想要的格式，以便于测试
处理成一行以“,”分隔的0-255的int值，width是70，那么长度就是14700，然后跟训练集做pca
'''

import cv2
import dlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from imutils.face_utils import FaceAligner
import argparse
import time

def main(imagePath,faceWidth,savePath):
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

    #np.savetxt(savePath, myImage, fmt="%d", delimiter=",")


    myImage = pd.DataFrame(myImage,columns=[str(i) for i in range(2700)])
    train=pd.read_csv("neoData/neo_set",names=[str(i) for i in range(2700)])

    print(myImage.info())
    print(train.info())
    train = pd.concat([train,myImage])
    pca = PCA(n_components=200)
    print(train.shape)
    train = pca.fit_transform(train)
    finalItem = train[train.shape[0]-1:]
    np.savetxt(savePath, finalItem,delimiter=",")
   # finalItem.to_csv(savePath,index=False,header=False)
    print("finish save",savePath)
    duration = time.time() - start_time
    print("Running %.3f sec All done!" % duration)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, default="parker1.jpg")
    parser.add_argument("--faceWidth", type=int, default=30, help="dlib_detect_face_width")
    parser.add_argument("--savePath", type=str, default='final_Item.csv')

    args = parser.parse_args()

    main(imagePath=args.imagePath, faceWidth=args.faceWidth, savePath=args.savePath)

