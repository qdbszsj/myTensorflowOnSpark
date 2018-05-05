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

# TensorFlow on spark：
先用spark-submit执行wiki_data_setup.py把数据存到hdfs里
然后执行wiki_spark.py训练，其中会调用wiki_dist.py


# 以上内容皆作废，尝试tf on spark 失败了，
# 后面我们用的spark mllib，pca加传统机器学习解决的。

# libraries Installation:
openCV:
sudo apt-get install python-opencv
dlib:
sudo apt-get install libboost-python-dev cmake
sudo pip install dlib
imutils:
sudo pip install imutils
numpy:
sudo pip install numpy
pandas:
Sudoku pip install pandas
sklearn:
sudo pip install sklearn


# download imdb and wiki dataset
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar
extract and put them into a folder named data

# extract features and labels
put the "shape_predictor_68_face_landmarks.dat" and scripts near the folder "data".
create a new folder named "myData"
run python scripts:
python imdb2csv.py --test_size 0.2 --face_width 30 --max_age 17 --min_age 16 --threadID 3
python imdb2csv.py --test_size 0.2 --face_width 30 --max_age 19 --min_age 18 --threadID 4
python imdb2csv.py --test_size 0.2 --face_width 30 --max_age 21 --min_age 20 --threadID 6
python imdb2csv.py --test_size 0.2 --face_width 30 --max_age 23 --min_age 22 --threadID 7
python wiki2csv.py --test_size 0.2 --face_width 30 --max_age 23 --min_age 22 --threadID 10
python wiki2csv.py --test_size 0.2 --face_width 30 --max_age 32 --min_age 30 --threadID 9
.......
these scripts can run on many machines and split the data according to age ranges.

rename the files generated in "myData" from ID 0 to n(up to you)

run python script:
wiki_mergeData.py 
before running, editing is needed.
change some parameters of this script: num_thread=n+1, imagePath="myData/", savePath="neoData/".
note that the script name is "wiki_.....", but it can also work on imdb, wiki and imdb are in same format after last step.
Then you can get a merged data in "neoData"



