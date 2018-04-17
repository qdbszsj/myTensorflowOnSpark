import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
pca = PCA(n_components=200)
train = pd.read_csv("neoData/neo_set",names=np.zeros(2700))
# pca
train = pca.fit_transform(train)
train = pd.DataFrame(train)
print(train.columns)
label = pd.read_csv('neoData/neo_setlabel',)
train['label'] = label['age']
# trans label to 0-2 "young" "middle" "old"
train.loc[train['label']<=25,'label'] = 0
train.loc[(train['label']>26) & (train['label']<=33),'label'] = 1
train.loc[(train['label']>34) & (train['label']<=50),'label'] = 2
#train.loc[(train['label']>25) & (train['label']<=30),'label'] = 3
#train.loc[(train['label']>30) & (train['label']<=40),'label'] = 4
#train.loc[(train['label']>40) & (train['label']<=50),'label'] = 5
#train.loc[(train['label']>50) & (train['label']<=60),'label'] = 6
train.loc[train['label']>50,'label'] = 4
# save data
train.to_csv("face_trainSet.csv",index=False,header=False)
