#Import the necessary modules to run pyspark, and the libraries used
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
import numpy as np

#Enter configuration
conf = SparkConf().setMaster("local").setAppName("LRG")

#Create spark context
sc = SparkContext(conf = conf)

#Load and examine the data. I remove any NA values.
path = "auto_mpg_original.csv"
raw_data = sc.textFile(path)
num_data = raw_data.count()
records = raw_data.map(lambda x: x.split(",")).filter(lambda r: 'NA' not in r)
first = records.first()
print first
print num_data

#Cache the RDD
records.cache()

#Function to extract features (all except car name and mpg). We don't need to use 1-of-k binary binding for the decision tree
def extract_features_dt(record):
    return np.array(map(float, record[1:7]))

#Function to extract target variable (mpg)
def extract_label(record):
    return float(record[0])

#Create LabeledPoint with displacement and horsepower
data_dt = records.map(lambda r: LabeledPoint(extract_label(r),extract_features_dt(r)))

#Print the extracted feature vector
first_point_dt = data_dt.first()
print "Decision Tree feature vector: " + str(first_point_dt.features)
print "Decision Tree feature vector length: " + str(len(first_point_dt.features))

#Split the dataset into training and testing for Decision Tree
data_with_idx = data_dt.zipWithIndex().map(lambda (k, v): (v, k))
test = data_with_idx.sample(False, 0.2, 42)
train = data_with_idx.subtractByKey(test)

train_data = train.map(lambda (idx, p): p)
test_data = test.map(lambda (idx, p) : p)
train_size = train_data.count()
test_size = test_data.count()
print "Training data size: %d" % train_size
print "Test data size: %d" % test_size
print "Total data size: %d " % num_data
print "Train + Test size : %d" % (train_size + test_size)

print "First 5 train data records: " + str(train_data.take(5))
print "First 5 test data records: " + str(test_data.take(5))

#Train the model from the train data
dt_model = DecisionTree.trainRegressor(train_data,{})
preds = dt_model.predict(test_data.map(lambda p: p.features))
actual = test_data.map(lambda p: p.label)
true_vs_predicted_dt = actual.zip(preds)

#Show a few predictions, depth and number of tree nodes
print "Decision Tree predictions: " + str(true_vs_predicted_dt.take(5))
print "Decision Tree depth: " + str(dt_model.depth())
print "Decision Tree number of nodes: " + str(dt_model.numNodes())


#Function to calculate absolute error
def abs_error(actual, pred):
    return np.abs(pred - actual)

#Function to calculate squared log error
def squared_log_error(pred, actual):
    return (np.log(pred + 1) - np.log(actual + 1))**2

#Calculate RMSLE and MAE and print the relevant values
rmsle_dt = np.sqrt(true_vs_predicted_dt.map(lambda (t, p): squared_log_error(t, p)).mean())
mae_dt = true_vs_predicted_dt.map(lambda (t, p): abs_error(t, p)).mean()

print "Decision Tree - Mean Absolute Error: %2.4f" % mae_dt
print "Decision Tree - Root Mean Squared Log Error: %2.4f" % rmsle_dt