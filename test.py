# Dependencies
# final script
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

'''
SPARK
'''

spark = SparkSession.builder.appName('RHODS').getOrCreate()

spark.catalog.clearCache()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 1024 * 1024 * 1024)

'''
DATA EXTRACTION
'''

# Connection to the PostgreSQL database
jdbc_url = "jdbc:postgresql://db-central-primary.sales-challenge.svc:5432/central"
connection_properties = {
    "user": "central",
    "password": "gs=@,(9;+zIYZY<k;b*87}na",
    # "driver": "org.postgresql.Driver"
}

# Store the data from the database tables to spark dataframes
main_df = spark.read.jdbc(url=jdbc_url, table="train", properties=connection_properties)
oil_df = spark.read.jdbc(url=jdbc_url, table="oil", properties=connection_properties)
holidays_df = spark.read.jdbc(url=jdbc_url, table="holidays_events", properties=connection_properties)
transactions_df = spark.read.jdbc(url=jdbc_url, table="transactions", properties=connection_properties)

spark.stop()