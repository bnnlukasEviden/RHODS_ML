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

print('data loaded')
#-------------------------------------------------------------------------------------------------------------------------------

'''
PREPROCESSING
'''
# # FEATURE ENGINEERING

# Aggregation of the main_df to get the sales volumes aggregatet over all stores for each product family and date
agg_df = main_df.groupBy('date', 'family').agg(sum('sales').alias("sales"), sum('onpromotion').alias("onpromotion"))

# Adding of features related to the day of the week and the month of each record based on the date
agg_df = agg_df.withColumn('day_of_week', dayofweek(agg_df.date)).withColumn('month', month(agg_df.date))


# Oil_df Merge
# agg_df and the oil_df are getting merged to use the oil price in the ML-Part of the project


# Aggregating the oil_df and the agg_df
agg_oil_merge_df = agg_df.join(oil_df, on='date', how='left') # left join agg_df and oil_df
agg_oil_merge_df = agg_oil_merge_df.dropDuplicates(['date']) # reduce df to one row per date




# --> Due to the high number of NaN-Values regarding the oilprice-value we decided to do a oilprice forecast
#-------------------------------------------------------------------------------------------------------------------------------

'''
Oil Price Forecasting
'''

# Create a subset of agg_oil_merge_df to forecast the oil prices
oil_forecast_df = agg_oil_merge_df.select('date', 'dcoilwtico', 'day_of_week', 'month')


n_lags = 5 # number of lags
window_spec = Window.orderBy('date')
for i in range(1, n_lags+1):
  oil_forecast_df = oil_forecast_df.withColumn("lag_{}".format(i), F.lag("dcoilwtico", offset = i).over(window_spec).cast('float'))

oil_forecast_df = oil_forecast_df.withColumn("dcoilwtico", oil_forecast_df["dcoilwtico"].cast('float'))


# Data transformation/encoding/split, Pipeline creation

numerical_cols_oil = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
categorical_cols_oil = ['day_of_week', 'month']

# Defining the encoder for encoding the categorical values
encoder_oil = OneHotEncoder(inputCols=categorical_cols_oil,
                        outputCols=[f"{col}_encoded" for col in categorical_cols_oil])

oil_forecast_encoded_df = encoder_oil.fit(oil_forecast_df).transform(oil_forecast_df)


# Create the train set of data with oil prices
oil_fc_train = oil_forecast_encoded_df.filter(oil_forecast_encoded_df['dcoilwtico'].isNotNull())
# Create the prediction set of data without oil prices
oil_fc_pred = oil_forecast_encoded_df.filter(oil_forecast_encoded_df['dcoilwtico'].isNull())


# Set the imputer
numerical_imputer_oil = Imputer(strategy="median", inputCols=numerical_cols_oil, outputCols=[f"{col}_imputed" for col in numerical_cols_oil])

# Assemble features into a single vector
feature_cols_oil = [f"{col}_imputed" for col in numerical_cols_oil] + [f"{col}_encoded" for col in categorical_cols_oil]
assembler_oil = VectorAssembler(inputCols=feature_cols_oil ,outputCol="features")

# Initialise the oilprice prediction model
rfr_oil = RandomForestRegressor(featuresCol='features', labelCol='dcoilwtico')

# Create the data preprocessing pipeline
pipeline_oil = Pipeline(stages=[numerical_imputer_oil, assembler_oil, rfr_oil]) # pipeline for cross validation

##### MODEL TRAINING #####

paramGrid_oil = ParamGridBuilder() \
    .addGrid(rfr_oil.maxDepth, [5,7]) \
    .addGrid(rfr_oil.numTrees, [10, 20, 30]) \
    .addGrid(rfr_oil.maxDepth, [5, 10, 15]) \
    .build()

evaluator_oil = RegressionEvaluator(labelCol="dcoilwtico", predictionCol="prediction", metricName="mae")

crossval_oil = CrossValidator(estimator=pipeline_oil,
                          estimatorParamMaps=paramGrid_oil,
                          evaluator=evaluator_oil,
                          numFolds=5)

cv_model_oil = crossval_oil.fit(oil_fc_train)

best_model_oil = cv_model_oil.bestModel # Model with the best parameters

oil_predictions = best_model_oil.transform(oil_fc_pred)


#  Oil price dataframe creation
oil_new_df = oil_fc_train.select('date', 'dcoilwtico').union(oil_predictions.select('date', 'prediction').withColumnRenamed("prediction",'dcoilwtico'))


# Now the new_oil_df gets merged with the agg_df to do further preprocessing
agg_oilprice_merged_df = agg_df.join(oil_new_df, on='date', how='left')



# ## Transactions_df Merge
# In this section we merge the aggregations_df with the agg_df to use the number of transaction per day for furter predictions

agg_transactions_df = transactions_df.groupBy("date").agg(sum("transactions").alias("transactions"))

# merging 
merged_df = agg_oilprice_merged_df.join(agg_transactions_df, on='date', how='left')
merged_df = merged_df.na.drop()


# ### Holidays_df Merge
# holidays_df gets merged to use the transaction data for the upcoming model prediction
# We only want to use the Holiday/Additional,etc. data which counts for the whole country Ecuador
modified_holidays_df = holidays_df.filter((col("transferred") == False) & (col("locale") == "National"))
modified_holidays_df = modified_holidays_df.withColumn("type", when(col("type") == "Transfer", "Holiday").otherwise(col("type")))
modified_holidays_df = modified_holidays_df.select("date", "type")

# Doing the merge of modified_holidays_df and merged_df
merged_df = merged_df.join(modified_holidays_df, on="date", how="left")
# The dates on which no holiday or something else encounted in the holiday_df takes place we insert "Normal" as a value
merged_df = merged_df.withColumn("type", when(merged_df["type"].isNull(), "Normal").otherwise(merged_df["type"]))

print('oil forecasting done')
#-------------------------------------------------------------------------------------------------------------------------------

'''
SALES FORECASTING
'''


'''
FEATURE-BASED (without time lags)
'''
# In the first trained model for predicting the sales volumes we use a normal feature based prediction without taking into account previous values.

# ## Data transformation/encoding
transformed_df_fb = merged_df

transformed_df_fb = transformed_df_fb.drop("date")

transformed_df_fb = transformed_df_fb.withColumnRenamed("day_of_week", "day_of_week_index")
transformed_df_fb = transformed_df_fb.withColumnRenamed("month", "month_index")
transformed_df_fb = transformed_df_fb.sample(withReplacement=False, fraction=0.05)

str_cat_cols_fb = ["type", "family"]
cat_cols_fb = ["day_of_week", "month", "type", "family"]

indexers_fb = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in str_cat_cols_fb]
encoders_fb = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_encoded") for col in cat_cols_fb]

for encoder in encoders_fb:
    encoder.setHandleInvalid("keep")
    encoder.setDropLast(True)

feature_cols_fb = ["day_of_week_encoded", "month_encoded", "type_encoded", "family_encoded", "transactions", "dcoilwtico", "onpromotion"]

##### MODEL TRAINING #####

assembler_fb = VectorAssembler(inputCols=feature_cols_fb, outputCol="features")

gbt_fb = GBTRegressor(featuresCol="features", labelCol="sales", maxBins=33)

pipeline_fb = Pipeline(stages= indexers_fb + encoders_fb + [assembler_fb, gbt_fb])

# Hyperparameter Tuning
paramGrid_fb = (ParamGridBuilder()
             .addGrid(gbt_fb.maxDepth, [4, 6])
             .addGrid(gbt_fb.maxIter, [50, 100])
             .addGrid(gbt_fb.stepSize, [0.1, 0.01])
             .build())

evaluator_fb = RegressionEvaluator(labelCol="sales", predictionCol="prediction", metricName="mae")

crossval_fb = CrossValidator(estimator=pipeline_fb,
                          estimatorParamMaps=paramGrid_fb,
                          evaluator=evaluator_fb,
                          numFolds=2)

train_data_fb, test_data_fb = transformed_df_fb.randomSplit([0.8, 0.2], seed=12)
cvModel_fb = crossval_fb.fit(train_data_fb)

'''
MODEL EVALUATION 
without time lags
'''

#  Overall evaluation


evaluation_fb_df = cvModel_fb.transform(test_data_fb)
clipped_evaluation_fb_df = evaluation_fb_df.withColumn("clipped_predictions", when(col("prediction") < 0, 0).otherwise(col("prediction")))

mae_fb = evaluator_fb.evaluate(clipped_evaluation_fb_df)


best_model_fb = cvModel_fb.bestModel


initial_types_fb = buildInitialTypesSimple(test_data_fb.drop("sales", 'date'))
onnx_model_fb = convert_sparkml(best_model_fb, 'Pyspark model with time lags', initial_types_tl, spark_session = spark)

with open("onnx_model.onnx", "wb") as onnx_file:
    onnx_file.write(onnx_model_fb.SerializeToString())

bucket_name_model = 'models'
object_key_model = 'model.onnx'  # You can adjust the path and name as needed.

try:
    s3.upload_file("./onnx_model.onnx", bucket_name_model, object_key_model)
    print(f"ONNX model uploaded to S3 bucket {bucket_name_model} with key {object_key_model}")
except NoCredentialsError:
    print("AWS credentials not available.")

#-------------------------------------------------------------------------------------------------------------------------------

spark.stop()