#encoding=utf-8
'''
这个要配合wiki2csv和wiki_mergeData来使用，先wiki2csv存到了myData文件夹下，如果有多个进程，存成好几份，要先运行wiki_mergeData.py合并成
train_set,test_set,train_label,testlabel
然后运行这个脚本，把四个文件合成一个文件。名为nei_set
用int表示的csv，即每一行都是用","隔开的int值。
第一列是label，一个1-100的int值
imageWidth是之前选取的图片宽度，如果有100张图片，宽度是20，那么读入的文件是一个长度为100*20*20*3的字符串
最终输出的csv文件是100行，每行是20*20*3共1200个用逗号隔开的0-255的int值，第一列再加一个label
'''
import numpy as np
import pandas as pd
###########################################################################
trainSetPath="train_set"
testSetPath="test_set"
trainLabelPath="train_label.csv"
testLabelPath="test_label.csv"
imageWidth=30
savePath="neo_set"
###########################################################

trainSetFile = open(trainSetPath,"rb")
trainSetString=trainSetFile.read()
testSetFile = open(testSetPath,"rb")
testSetString=testSetFile.read()
imagesString=str(trainSetString)+str(testSetString)
totalLen=len(imagesString)
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
#labels=label.values
#total=np.c_[labels,images]

print(images.shape)#m * 76800

np.savetxt(savePath, images, fmt="%d", delimiter=",")
label.to_csv(savePath+'label')


print("finish save "+ savePath)
