from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

def toCSV(vec):
  """Converts a vector/array into a CSV string"""
  return ','.join([str(i) for i in vec])


def fromCSV(s):
  """Converts a CSV string to a vector/array"""
  return [float(x) for x in s.split(',') if len(s) > 0]


def writeWiki(sc, imagePath, labelPath, output, num_partitions, imageWidth):
  """Writes wiki image/label vectors into parallelized files on HDFS"""
  # load wiki into memory
  
  imagesFile = open(imagePath,"rb") 
  imagesString=imagesFile.read() 
  totalLen=len(imagesString)
  singleLen=imageWidth*imageWidth*3
  number=int(totalLen/singleLen)
  print(totalLen,singleLen,number)
  imageList=[]
  for i in range(totalLen):
    imageList.append(imagesString[i])
  images=np.array(imageList).reshape(number, singleLen)
  

  labelFile=pd.read_csv(labelPath, delimiter=",")
  labels=np.array(labelFile['age'].values[:]).reshape(number,1)
  print(images.shape)#m * 76800
  print(labels.shape)#m * 1

  # create RDDs of vectors
  imageRDD = sc.parallelize(images, num_partitions)
  labelRDD = sc.parallelize(labels, num_partitions)

  output_images = output + "/images"
  output_labels = output + "/labels"

  # save RDDs as specific format
  imageRDD.map(toCSV).saveAsTextFile(output_images)
  labelRDD.map(toCSV).saveAsTextFile(output_labels)






if __name__ == "__main__":
  import argparse

  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf

  parser = argparse.ArgumentParser()
  parser.add_argument("--num-partitions", help="Number of output partitions", type=int, default=10)
  parser.add_argument("--output", help="HDFS directory to save examples in parallelized format", default="wiki_data")
  parser.add_argument("--verify", help="verify saved examples after writing", action="store_true")
  parser.add_argument("--trainImagePath", help="pathOfTrainImage", type=str, default="wiki/train_set")
  parser.add_argument("--trainLabelPath", help="pathOfTrainLabelCSV", type=str, default="wiki/train_label.csv")
  parser.add_argument("--testImagePath", help="pathOfTestImage", type=str, default="wiki/test_set")
  parser.add_argument("--testLabelPath", help="pathOfTestLabelCSV", type=str, default="wiki/test_label.csv")
  parser.add_argument("--image_width", help="Number of output partitions", type=int, default=50)

  args = parser.parse_args()
  print("args:", args)

  sc = SparkContext(conf=SparkConf().setAppName("wiki_parallelize"))
  sc=1

  writeWiki(sc, args.trainImagePath, args.trainLabelPath, args.output + "/train", args.num_partitions, args.image_width)
  writeWiki(sc, args.testImagePath, args.testLabelPath, args.output + "/test", args.num_partitions, args.image_width)

