'''
把wiki2csv的train_set或者test_set转化成用int表示的csv，即每一行都是用","隔开的int值。
imageWidth是之前选取的图片宽度，如果有100张图片，宽度是20，那么读入的文件是一个长度为100*20*20*3的字符串
最终输出的csv文件是100行，每行是20*20*3共1200个用逗号隔开的0-255的int值
'''
import numpy as np
#这里参数设置，很重要
###########################################################################
imagePath="train_set"
imageWidth=50
savePath="neo_train_set"
###########################################################

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

print(images.shape)#m * 7500

np.savetxt('neo_train_set', images, fmt="%d", delimiter=",")

imagesFile.close()

print("finish save "+ savePath)
