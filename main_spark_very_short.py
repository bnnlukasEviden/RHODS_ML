from io import StringIO
import pandas as pd
import io

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the log level as needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql.functions import sum, when, avg, max, col, dayofweek, count, month, lit, mean
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import expr
from pyspark.sql import Row
from pyspark.sql.types import DateType

from onnxmltools import convert_sparkml
from onnxmltools.convert.sparkml.utils import buildInitialTypesSimple

import boto3
from botocore.exceptions import NoCredentialsError

pd.DataFrame.iteritems = pd.DataFrame.items

ionos_endpoint_url = 'https://sales-challenge.s3-eu-central-1.ionoscloud.com'
ionos_region = 'de'

ionos_access_key = '00d4804864744cc11eb4'
ionos_secret_key = 'peqgMLZSwt59HRwg+BgAMxeuwDkkIwkYcTVRSZWq'

s3 = boto3.client('s3',
    aws_access_key_id=ionos_access_key,
    aws_secret_access_key=ionos_secret_key,
    endpoint_url=ionos_endpoint_url,
    region_name=ionos_region
)

spark = SparkSession.builder.appName('RHODS').getOrCreate()

spark.catalog.clearCache()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

bucket_name_csv = 'csv'

bucket_name = 'csv'
object_key = 'train.csv'
csv_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
body = csv_obj['Body']
csv_string = body.read().decode('utf-8')
main_df = spark.createDataFrame(pd.read_csv(StringIO(csv_string)))

main_df = main_df.limit(30)
main_df_cut = main_df.select('store_nbr', 'onpromotion', 'sales')
feature_cols = ["store_nbr", 'onpromotion']

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

gbt = GBTRegressor(featuresCol="features", labelCol="sales")

pipeline = Pipeline(stages= [assembler, gbt])

# Hyperparameter Tuning
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [4])
             .addGrid(gbt.maxIter, [50])
             .addGrid(gbt.stepSize, [0.1])
             .build())

evaluator = RegressionEvaluator(labelCol="sales", predictionCol="prediction", metricName="mae")

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2)

train_data, test_data = main_df_cut.randomSplit([0.8, 0.2], seed=12)
cvModel = crossval.fit(train_data)

best_model = cvModel.bestModel

predictions = best_model.transform(test_data)

initial_types = buildInitialTypesSimple(test_data.drop("sales"))

logging.info(f"The value of my_variable is: {predictions}")
onnx_model = convert_sparkml(best_model, 'Pyspark model without time lags', initial_types, spark_session = spark)


onnx_bytes = onnx_model.SerializeToString()

bucket_name_model = 'models'
object_key_model = 'model_test_pipeline3.onnx'

try:
    s3.put_object(Bucket = bucket_name_model, Key = object_key_model, Body = onnx_bytes)
    print(f"ONNX model uploaded to S3 bucket {bucket_name_model} with key {object_key_model}")
except NoCredentialsError:
    print("AWS credentials not available.")

spark.stop()