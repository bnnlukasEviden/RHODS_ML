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
from pyspark.sql.functions import sum, when, avg, max, col, dayofweek, count, month, lit, mean

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

import openvino as ov

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

# Aggregation of the main_df to get the sales volumes aggregatet over all stores for each product family and date
agg_df = main_df.groupBy('date', 'family').agg(sum('sales').alias("sales"), sum('onpromotion').alias("onpromotion"))

# Adding of features related to the day of the week and the month of each record based on the date
agg_df = agg_df.withColumn('day_of_week', dayofweek(agg_df.date)).withColumn('month', month(agg_df.date))
agg_df.sort(col("date"), col("family")).show(truncate = False)

agg_df_pd = agg_df.toPandas()

grouped = agg_df_pd.groupby('family')
family_dataframes = {name: group for name, group in grouped}

automotive_df = family_dataframes['AUTOMOTIVE']
modified_automotive_df = automotive_df[['date', 'sales']]

window_size = 10

X, y = [], []
for i in range(len(modified_automotive_df) - window_size):
    try:
        X.append(modified_automotive_df['sales'].iloc[i:i+window_size].values)
        y.append(modified_automotive_df['sales'].iloc[i+window_size])
    except:
        if len(X) != len(y):
            X.pop()
        break
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = keras.Sequential()
model.add(keras.layers.LSTM(50, activation='relu', input_shape=(window_size, 1)))
model.add(keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mean_absolute_error')

model.fit(X_train, y_train, epochs=1, batch_size=16)

tf.saved_model.save(model,'model')
ov_model = ov.convert_model('./model')

ov.save_model(ov_model, 'model.xml')

bucket_name_model = 'models'
object_key_model = 'lstm_model.xml'

try:
    s3.upload_file('./model.xml', bucket_name_model, object_key_model)
    print(f"ONNX model uploaded to S3 bucket {bucket_name_model} with key {object_key_model}")
except NoCredentialsError:
    print("AWS credentials not available.")

spark.stop()