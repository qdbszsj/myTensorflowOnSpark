# myTensorflowOnSpark

# 数据处理！！！！！
假设现在的工作目录是~/,训练时，先把wiki的图片下载下来，
data/wiki_crop/00/xxx.jpg
data/wiki_crop/00/xxy.jpg
data/wiki_crop/01/xxy.jpg
......

然后在“根目录”下运行
wiki2csv.py
语句大概是这样的，详情看脚本文件顶端的注释
python wiki2csv.py --test_size 0.4 --face_width 50 --max_age 5 --min_age 1 --threadID 0
python wiki2csv.py --test_size 0.4 --face_width 50 --max_age 10 --min_age 6 --threadID 1
......

然后会在myData/生成很多文件

之后执行
wiki_mergeData.py
在脚本里调参数，生成4个文件在neoData/里
分别是
train_set
train_label.csv
test_set
test_label.csv

然后这四个文件可以用来feed到tensorflow on spark 里，如果想要oneHot编码的label，还可以在这四个文件的同一目录里执行
wiki_label2oneHot.py

另外，Neo希望把四个文件合并到一起，第一列是label，后面是int型的data，于是我们可以执行
wiki_data_Transfer_ForNeo.py
会在同一目录下生成neo_set这个文件






#TensorFlow on spark：
先用spark-submit执行wiki_data_setup.py把数据存到hdfs里
然后执行wiki_spark.py训练，其中会调用wiki_dist.py
