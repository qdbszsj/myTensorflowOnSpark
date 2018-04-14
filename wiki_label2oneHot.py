import numpy as np
import pandas as pd

labelPath="train_label.csv"
savePath="train_label_oneHot.csv"


label=pd.read_csv(labelPath, delimiter=",")
label["1"]=0
label["2"]=0
label["3"]=0

value=label.values
m,n=value.shape

for i in range(m):
    if value[i,0]<=20:value[i,1]=1
    elif value[i,0]<=50:value[i,2]=1
    else:value[i,3]=1

labelSave = pd.DataFrame(data=value, columns=['age','1','2','3'])
label.to_csv(savePath,header=True,index=False, sep=",")
