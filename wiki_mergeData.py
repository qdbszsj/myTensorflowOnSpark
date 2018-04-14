
#######################
num_thread=6
imagePath="myData/"
savePath="neoData/"
##############################
import numpy as np
import pandas as pd


def mergeDataset(datasetName):
    F = open(savePath+datasetName,"w") 
    for i in range(num_thread):
        imagesFile = open(imagePath+datasetName+str(i),"rb") 
        imagesString=imagesFile.read() 
        print("write"+datasetName,len(imagesString))
        F.write(imagesString)
        imagesFile.close()
    F.close()

mergeDataset("train_set")
mergeDataset("test_set")



def mergeLabel(labelName): 
    label=pd.read_csv(imagePath+labelName+"0")
    for i in range(1,num_thread):
        curLabel=pd.read_csv(imagePath+labelName+str(i))
        label=pd.concat([label,curLabel])
    label['age'].to_csv(savePath+labelName+".csv",header=True,index=False, sep=",")
    print(label.shape)


print("finish save "+ savePath)
