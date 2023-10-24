# Imports



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




spark = SparkSession.builder.appName('RHODS').getOrCreate()




spark.catalog.clearCache()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 1024 * 1024 * 1024)


'''
DATA EXTRACTION
'''


main_df = spark.read.csv('train.csv', header = True, inferSchema = True)
oil_df = spark.read.csv('oil.csv', header = True, inferSchema = True)
holidays_df = spark.read.csv('holidays_events.csv', header = True, inferSchema = True)
transactions_df = spark.read.csv('transactions.csv', header = True, inferSchema = True)
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

# Calculate the percentage of NaN values in the 'dcoilwtico' column
percentage_nan = (agg_oil_merge_df.filter(col("dcoilwtico").isNull()).count() / agg_oil_merge_df.count()) * 100
print(f'Procentual number of NaN-Values: {percentage_nan}%')


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

numerical_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
categorical_cols = ['day_of_week', 'month']

# Defining the encoder for encoding the categorical values
encoder = OneHotEncoder(inputCols=categorical_cols,
                        outputCols=[f"{col}_encoded" for col in categorical_cols])

oil_forecast_encoded_df = encoder.fit(oil_forecast_df).transform(oil_forecast_df)



# Create the train set of data with oil prices
oil_fc_train = oil_forecast_encoded_df.filter(oil_forecast_encoded_df['dcoilwtico'].isNotNull())
# Create the prediction set of data without oil prices
oil_fc_pred = oil_forecast_encoded_df.filter(oil_forecast_encoded_df['dcoilwtico'].isNull())


# Set the imputer
numerical_imputer = Imputer(strategy="median", inputCols=numerical_cols, outputCols=[f"{col}_imputed" for col in numerical_cols])

# Assemble features into a single vector
feature_cols = [f"{col}_imputed" for col in numerical_cols] + [f"{col}_encoded" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols,outputCol="features")

# Initialise the oilprice prediction model
rfr_oil = RandomForestRegressor(featuresCol='features', labelCol='dcoilwtico')

# Create the data preprocessing pipeline
oil_pipeline = Pipeline(stages=[numerical_imputer, assembler, rfr_oil]) # pipeline for cross validation

##### MODEL TRAINING #####

param_grid = ParamGridBuilder() \
    .addGrid(rfr_oil.maxDepth, [5,7]) \
    .addGrid(rfr_oil.numTrees, [10, 20, 30]) \
    .addGrid(rfr_oil.maxDepth, [5, 10, 15]) \
    .build()

evaluator = RegressionEvaluator(labelCol="dcoilwtico", predictionCol="prediction", metricName="mae")

crossval = CrossValidator(estimator=oil_pipeline,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=5)

cv_model_oil = crossval.fit(oil_fc_train)

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
modified_holidays_df.sort(col("date")).show(truncate = False)

# Doing the merge of modified_holidays_df and merged_df
merged_df = merged_df.join(modified_holidays_df, on="date", how="left")
# The dates on which no holiday or something else encounted in the holiday_df takes place we insert "Normal" as a value
merged_df = merged_df.withColumn("type", when(merged_df["type"].isNull(), "Normal").otherwise(merged_df["type"]))
merged_df.sort(col("date"), col('family')).show(truncate = False)
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
transformed_df_fb = transformed_df_fb.withColumnRenamed("day_of_week", "day_of_week_index")
transformed_df_fb = transformed_df_fb.withColumnRenamed("month", "month_index")

str_cat_cols_fb = ["type", "family"]
cat_cols_fb = ["day_of_week", "month", "type", "family"]

indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in str_cat_cols_fb]
encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_encoded") for col in cat_cols_fb]

pipeline = Pipeline(stages=indexers + encoders)
model = pipeline.fit(transformed_df_fb)
data = model.transform(transformed_df_fb)
data.sort(col("date"), col('family')).show(truncate = False)


##### MODEL TRAINING #####

feature_cols = ["day_of_week_encoded", "month_encoded", "type_encoded", "family_encoded", "transactions", "dcoilwtico", "onpromotion"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
gbt = GBTRegressor(featuresCol="features", labelCol="sales", maxBins=33) # Gradient Boosting Regressor
pipeline = Pipeline(stages=[assembler, gbt])


# Hyperparameter Tuning
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [4, 6])
             .addGrid(gbt.maxIter, [50, 100])
             .addGrid(gbt.stepSize, [0.1, 0.01])
             .build())


evaluator = RegressionEvaluator(labelCol="sales", predictionCol="prediction", metricName="mae")

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2)

train_data, test_data = data.randomSplit([0.8, 0.2], seed=12)
cvModel = crossval.fit(train_data)


'''
MODEL EVALUATION 
without time lags
'''

#  Overall evaluation


evaluation_fb_df = cvModel.transform(test_data)
clipped_evaluation_fb_df = evaluation_fb_df.withColumn("clipped_predictions", when(col("prediction") < 0, 0).otherwise(col("prediction")))

mae = evaluator.evaluate(clipped_evaluation_fb_df)

best_model = cvModel.bestModel

print("Mean Absolute Error (MAE):", mae)



#  Product family wise evaluation

filtered_evaluation_fb_df = clipped_evaluation_fb_df.select("family", "sales", "clipped_predictions")
evaluator = RegressionEvaluator(labelCol="sales", predictionCol="clipped_predictions", metricName="mae")
unique_family_values = filtered_evaluation_fb_df.select("family").distinct().rdd.flatMap(lambda x: x).collect()

data = []
for value in unique_family_values:
  df_fb = filtered_evaluation_fb_df.filter(filtered_evaluation_fb_df['family'] == value)
  mae_fb = evaluator.evaluate(df_fb)
  mean_fb = df_fb.select(mean("sales")).collect()[0][0]
  row = Row(family=value, mean=mean_fb, mean_absolute_error=mae_fb)
  data.append(row)

family_evaluation_fb_df = spark.createDataFrame(data)


tl_df = merged_df # creating plain df for time-lagged data frame

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

indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in str_cat_cols_tl]
encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_encoded") for col in cat_cols_tl]

pipeline = Pipeline(stages=indexers + encoders)
model = pipeline.fit(transformed_df_tl)
data = model.transform(transformed_df_tl)
data.sort(col("date"), col('family')).show(truncate = False)
#-------------------------------------------------------------------------------------------------------------------------------

'''
TIME-LAG MODEL
'''


feature_cols = ["day_of_week_encoded", "month_encoded", "type_encoded", "family_encoded", "transactions", "dcoilwtico", "onpromotion", "lag_1", "lag_2", "lag_3"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

gbt = GBTRegressor(featuresCol="features", labelCol="sales", maxBins=33)

pipeline = Pipeline(stages=[assembler, gbt])

paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [4, 6])
             .addGrid(gbt.maxIter, [50, 100])
             .addGrid(gbt.stepSize, [0.1, 0.01])
             .build())

evaluator = RegressionEvaluator(labelCol="sales", predictionCol="prediction", metricName="mae")

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2)

train_data, test_data = data.randomSplit([0.8, 0.2], seed=12)
cvModel = crossval.fit(train_data)


'''
EVALUATION
time-lagged model
'''

# Overall evaluation


evaluation_tl_df = cvModel.transform(test_data)
clipped_evaluation_tl_df = evaluation_tl_df.withColumn("clipped_predictions", when(col("prediction") < 0, 0).otherwise(col("prediction")))

mae_tl = evaluator.evaluate(evaluation_tl_df)

best_model_tl = cvModel.bestModel

print("Mean Absolute Error (MAE):", mae_tl)




# ### Product family wise evaluation


filtered_evaluation_fb_df = clipped_evaluation_tl_df.select("family", "sales", "clipped_predictions")
evaluator = RegressionEvaluator(labelCol="sales", predictionCol="clipped_predictions", metricName="mae")
unique_family_values = filtered_evaluation_fb_df.select("family").distinct().rdd.flatMap(lambda x: x).collect()

data = []
for value in unique_family_values:
  df_fb = filtered_evaluation_fb_df.filter(filtered_evaluation_fb_df['family'] == value)
  mae_fb = evaluator.evaluate(df_fb)
  mean_fb = df_fb.select(mean("sales")).collect()[0][0]
  row = Row(family=value, mean=mean_fb, mean_absolute_error=mae_fb)
  data.append(row)

family_evaluation_fb_df = spark.createDataFrame(data)
#-------------------------------------------------------------------------------------------------------------------------------

