#encoding=utf-8
'''
把wiki2csv的train_set或者test_set转化成用int表示的csv，即每一行都是用","隔开的int值。
imageWidth是之前选取的图片宽度，如果有100张图片，宽度是20，那么读入的文件是一个长度为100*20*20*3的字符串
最终输出的csv文件是100行，每行是20*20*3共1200个用逗号隔开的0-255的int值
'''
import numpy as np
import pandas as pd
#这里参数设置，很重要
###########################################################################
trainSetPath="train_set"
testSetPath="test_set"
trainLabelPath="train_label.csv"
testLabelPath="test_label.csv"
imageWidth=160
savePath="neo_set"
###########################################################

trainSetFile = open(trainSetPath,"rb") 
trainSetString=trainSetFile.read() 
testSetFile = open(testSetPath,"rb") 
testSetString=testSetFile.read()
imagesString=str(trainSetString)+str(testSetString)
totalLen=len(imageString)
print(totalLen)
trainSetFile.close()
testSetFile.close()

trainLabel=pd.read_csv(trainLabelPath)
testLabel=pd.read_csv(testLabelPath)
label=pd.concat([trainLabel,testLabel])

singleLen=imageWidth*imageWidth*3
number=int(totalLen/singleLen)
print(totalLen,singleLen,number)
imageList=[]
for i in range(totalLen):
  imageList.append(ord(imagesString[i]))
images=np.array(imageList).reshape(number, singleLen)
images.astype(int)
labels=label.values
total=np.c_[labels,images]

print(total.shape)#m * 76800

np.savetxt(savePath, total, fmt="%d", delimiter=",")


print("finish save "+ savePath)
