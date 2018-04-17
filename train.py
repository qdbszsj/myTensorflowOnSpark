# by neo 2018-04-11 18:27:40

from __future__ import print_function
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
# $example on$
from pyspark.ml.classification import LogisticRegression
# $example off$
from pyspark.sql import SparkSession
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
from pyspark.streaming import StreamingContext
from time import *
if __name__ == "__main__":
    RANDOM_SEED = 13579
    TRAINING_DATA_RATIO = 0.9
    RF_NUM_TREES = 100
    RF_MAX_DEPTH = 4
    RF_NUM_BINS = 32
    spark = SparkSession\
        .builder\
        .appName("RF_classification")\
        .getOrCreate()
    ssc = StreamingContext(spark, 1)
    #df = spark.read.option("header","false").csv("hdfs://student61:9000/wiki/face_trainSet.csv")
    df = spark.read.option("header","false").option("inferSchema","true").csv("hdfs://student61:9000/wiki/face_trainSet.csv")

    print('Finished reading Data.........')
    transformed_df = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:-1])))
    splits = [TRAINING_DATA_RATIO, 1.0 - TRAINING_DATA_RATIO]
    training_data, test_data = transformed_df.randomSplit(splits, RANDOM_SEED)
    print("Number of training set rows: %d" % training_data.count())
    print("Number of test set rows: %d" % test_data.count())
    start_time = time()
    #model = GradientBoostedTrees.trainClassifier(training_data,
    #                                             categoricalFeaturesInfo={}, numIterations=30)
    # model = RandomForest.trainClassifier(training_data, numClasses=8, categoricalFeaturesInfo={}, \
    #      numTrees=RF_NUM_TREES, featureSubsetStrategy="auto", impurity="gini", \
    #      maxDepth=RF_MAX_DEPTH, maxBins=RF_NUM_BINS, seed=RANDOM_SEED)
    model = GradientBoostedTrees.trainRegressor(training_data,
                                            categoricalFeaturesInfo={}, numIterations=100)
    end_time = time()
    elapsed_time = end_time - start_time
    print("---------------------------------------------------")
    print("Time to train RF: %.3f seconds" % elapsed_time)
    print("---------------------------------------------------")
    # Evaluate model on test instances and compute test error
    predictions = model.predict(test_data.map(lambda x: x.features))
    labelsAndPredictions = test_data.map(lambda lp: lp.label).zip(predictions)
    testMSE = labelsAndPredictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum()/float(test_data.count())
    print("---------------------------------------------------")
    print('Test Mean Squared Error = ' + str(testMSE))
    print('Learned regression GBT model:')
    print(model.toDebugString())
    print("---------------------------------------------------")
    # use Logistic Regression
    # start_time = time()
    # LR_model = LogisticRegressionWithSGD.train(training_data, 100)
    # end_time = time()
    # elapsed_time = end_time - start_time
    # print("---------------------------------------------------")
    # print("Time to train LR: %.3f seconds" % elapsed_time)
    # print("---------------------------------------------------")
    # save model
    model.save(spark.sparkContext,"hdfs://student61:9000/wiki/GBDT_regression_model")
    print("---------------------------------------------------")
    print("succesfully save the model")
    print("---------------------------------------------------")
    spark.stop()
