# by neo 2018-04-11 18:27:40

from __future__ import print_function
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.sql import SparkSession
from pyspark.mllib.tree import RandomForest,RandomForestModel
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.streaming import StreamingContext
from time import *
if __name__ == "__main__":
    RANDOM_SEED = 13579
    TRAINING_DATA_RATIO = 0.9
    RF_NUM_TREES = 3
    RF_MAX_DEPTH = 4
    RF_NUM_BINS = 32
    spark = SparkSession\
        .builder\
        .appName("predict")\
        .getOrCreate()
    # spark streaming initialization
    #df = spark.read.option("header","false").csv("hdfs://student61:9000/wiki/final_Item.csv")
    df = spark.read.option("header","false").option("inferSchema","true").csv("hdfs://student61:9000/wiki/final_Item.csv")

    #df.cast(DoubleType)
    print('parker!!!',df)
    print(df.first())
    df.show()
    test_data = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:])))
    print('testData!!!',test_data)
    # load model
    start_time = time()
    #model = GradientBoostedTreesModel.load(spark.sparkContext,"hdfs://student61:9000/wiki/GBDT_model")
   # model = RandomForestModel.load(spark.sparkContext,"hdfs://student61:9000/wiki/RF_model")
    model = GradientBoostedTreesModel.load(spark.sparkContext,"hdfs://student61:9000/wiki/GBDT_regression_model")
    #print(model.toDebugString())
    end_time = time()
    elapsed_time = end_time - start_time
    print("---------------------------------------------------")
    print("Time to load model: %.3f seconds" % elapsed_time)
    print("---------------------------------------------------")
    # make predictions
    predictions = model.predict(test_data.map(lambda x: x.features))
    end_time = time()
    elapsed_time = end_time - start_time
    print("---------------------------------------------------")
    print("Time from load model to predictions: %.3f seconds" % elapsed_time)
    print("---------------------------------------------------")
    print('--------------------------------------------------------------------')
    print(predictions.top(10))
    preResult = str(predictions.top(1)[0])
    print('result is',preResult)
    print('--------------------------------------------------------------------')
    F=open('/opt/finalResult.txt','w')
    F.write(preResult)
    F.close()
    # predictions = rf.predict(test_data.map(lambda x: x.features))
    # labels_and_predictions = test_data.map(lambda x: x.label).zip(predictions)
    # acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
    # print('--------------------------------------------------------------------')
    # print("Model accuracy: %.3f%%" % (acc * 100))
    # print('--------------------------------------------------------------------')
    spark.stop()
