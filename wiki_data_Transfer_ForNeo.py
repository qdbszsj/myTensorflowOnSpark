


import pandas as pd
import numpy as np
'''
train=pd.read_csv('test_set.csv', delimiter=",")
m,n=train.shape
print(m,n)
'''

#F = open("train_set","rb") 
#train_set=F.read() 

#train=pd.read_csv('train_label.csv', delimiter=",")

#print(ord(all[1]))



imagePath="train_set"
imageWidth=50
#labelPath="train_label.csv"
savePath="neo_train_set"

imagesFile = open(imagePath,"rb") 
imagesString=imagesFile.read() 
totalLen=len(imagesString)
print(totalLen)

singleLen=imageWidth*imageWidth*3
number=int(totalLen/singleLen)
print(totalLen,singleLen,number)
imageList=[]
for i in range(totalLen):
  imageList.append(ord(imagesString[i]))
images=np.array(imageList).reshape(number, singleLen)
images.astype(int)

#labelFile=pd.read_csv(labelPath, delimiter=",")
#labels=np.array(labelFile['age'].values[:]).reshape(number,1)

print(images.shape)#m * 76800
#print(labels.shape)#m * 1


np.savetxt('neo_train_set', images, fmt="%d", delimiter=",")

imagesFile.close()

print("finish save "+ savePath)
