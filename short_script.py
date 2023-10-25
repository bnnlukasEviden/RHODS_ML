# Dependencies
# final script

from io import StringIO
import pandas as pd

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


'''
SPARK
'''

spark = SparkSession.builder.appName('RHODS').getOrCreate()

spark.catalog.clearCache()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 1024 * 1024 * 1024)

'''
DATA EXTRACTION
'''

bucket_name_csv = 'csv'

file_names = ['train', 'oil', 'holidays_events','transactions']
Data = {}

for x in file_names:
    bucket_name = 'csv'
    object_key = x + '.csv'
    csv_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    df = spark.createDataFrame(pd.read_csv(StringIO(csv_string)))
    Data[x] = df

main_df = Data['train']
oil_df = Data['oil']
holidays_df = Data['holidays_events']
transactions_df = Data['transactions']

#-------------------------------------------------------------------------------------------------------------------------------

'''
PREPROCESSING
'''
# # FEATURE ENGINEERING

# Aggregation of the main_df to get the sales volumes aggregatet over all stores for each product family and date
agg_df = main_df.groupBy('date', 'family').agg(sum('sales').alias("sales"), sum('onpromotion').alias("onpromotion"))

# Adding of features related to the day of the week and the month of each record based on the date
agg_df = agg_df.withColumn('day_of_week', dayofweek(agg_df.date)).withColumn('month', month(agg_df.date))





'''
SALES FORECASTING
'''



# SHORTENING DATA FRAME
tl_df = agg_df.limit(3000).where(col("sales").isNotNull())



#-------------------------------------------------------------------------------------------------------------------------------

'''
FEATURE-BASED WITH TIME LAG
'''



window_spec = Window.partitionBy("family").orderBy("date")

# adding lags 
tl_df = tl_df.withColumn("lag_1", F.lag("sales", 1).over(window_spec))
tl_df = tl_df.withColumn("lag_2", F.lag("sales", 2).over(window_spec))
tl_df = tl_df.withColumn("lag_3", F.lag("sales", 3).over(window_spec))


specific_date = "2013-01-04" # The Time lag dataframe should start from the "2013-01-04" because else there is no lag data for the first row
specific_date = spark.createDataFrame([(specific_date,)], ["specific_date"]).withColumn("specific_date", col("specific_date").cast(DateType()))
tl_df_filtered = tl_df.filter(col("date") >= specific_date.select("specific_date").collect()[0][0])


# Data transformation/encoding

transformed_df_tl = tl_df_filtered

transformed_df_tl = transformed_df_tl.withColumnRenamed("day_of_week", "day_of_week_index")
transformed_df_tl = transformed_df_tl.withColumnRenamed("month", "month_index")

str_cat_cols_tl = ["type", "family"]
cat_cols_tl = ["day_of_week", "month", "type", "family"]

indexers_tl = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in str_cat_cols_tl]
encoders_tl = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_encoded") for col in cat_cols_tl]

for encoder in encoders_tl:
    encoder.setHandleInvalid("keep")
    encoder.setDropLast(True)


transformed_df_tl = transformed_df_tl.withColumn("row_num", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
transformed_df_tl = transformed_df_tl.select('sales',"lag_1", "lag_2", "lag_3").filter(transformed_df_tl.row_num > 3)


##### MODEL TRAINING #####

feature_cols_tl = ["lag_1", "lag_2", "lag_3"]

assembler_tl = VectorAssembler(inputCols=feature_cols_tl, outputCol="features")

gbt_tl = GBTRegressor(featuresCol="features", labelCol="sales", maxBins=33)

pipeline_tl = Pipeline(stages= [assembler_tl, gbt_tl])

paramGrid_tl = (ParamGridBuilder()
             .addGrid(gbt_tl.maxDepth, [4])
             .addGrid(gbt_tl.maxIter, [50])
             .addGrid(gbt_tl.stepSize, [0.1])
             .build())

evaluator_tl = RegressionEvaluator(labelCol="sales", predictionCol="prediction", metricName="mae")

crossval_tl = CrossValidator(estimator=pipeline_tl,
                          estimatorParamMaps=paramGrid_tl,
                          evaluator=evaluator_tl,
                          numFolds=2)

train_data_tl, test_data_tl = transformed_df_tl.randomSplit([0.8, 0.2], seed=12)


# Null values rausfiltern (mir unklar wieso hier überhaupt noch welche drin sind)
train_data_tl = train_data_tl.where(col("sales").isNotNull()).where(col("lag_1").isNotNull()).where(col("lag_2").isNotNull()).where(col("lag_3").isNotNull())
test_data_tl =test_data_tl.where(col("sales").isNotNull()).where(col("lag_1").isNotNull()).where(col("lag_2").isNotNull()).where(col("lag_3").isNotNull())


cvModel_tl = crossval_tl.fit(train_data_tl)


'''
EVALUATION
time-lagged model
'''

evaluation_tl_df = cvModel_tl.transform(test_data_tl)
clipped_evaluation_tl_df = evaluation_tl_df.withColumn("clipped_predictions", when(col("prediction") < 0, 0).otherwise(col("prediction")))

mae_tl = evaluator_tl.evaluate(evaluation_tl_df)

best_model_tl = cvModel_tl.bestModel

initial_types_tl = buildInitialTypesSimple(test_data_tl.drop("sales", 'date'))
onnx_model_tl = convert_sparkml(best_model_tl, 'Pyspark model with time lags', initial_types_tl, spark_session = spark)

with open("onnx_model.onnx", "wb") as onnx_file:
    onnx_file.write(onnx_model_tl.SerializeToString())

bucket_name_model = 'models'
object_key_model = 'model.onnx'  # You can adjust the path and name as needed.

try:
    s3.upload_file("./onnx_model.onnx", bucket_name_model, object_key_model)
    print(f"ONNX model uploaded to S3 bucket {bucket_name_model} with key {object_key_model}")
except NoCredentialsError:
    print("AWS credentials not available.")

#-------------------------------------------------------------------------------------------------------------------------------

spark.stop()