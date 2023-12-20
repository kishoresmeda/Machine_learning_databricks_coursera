# Databricks notebook source
# Import necessary libraries and create a Spark session
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Airbnb Price Prediction") \
    .getOrCreate()

# Dataset downloaded from https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

df = spark.read.csv("dbfs:/FileStore/shared_uploads/courserabricks@outlook.com/AB_NYC_2019.csv", header=True, inferSchema=True)
display(df)

# COMMAND ----------

# Fix datatypes in the columns

from pyspark.sql.functions import col

columns = df.columns
col_data_types = {'id':'double',
 'name':'string',
 'host_id':'double',
 'host_name':'string',
 'neighbourhood_group':'string',
 'neighbourhood':'string',
 'latitude':'float',
 'longitude':'float',
 'room_type':'string',
 'price':'int',
 'minimum_nights':'int',
 'number_of_reviews':'int',
 'last_review':'string',
 'reviews_per_month':'float',
 'calculated_host_listings_count':'int',
 'availability_365':'int'}

# change the datatype of each column
for col_name, col_data_type in col_data_types.items():
    df = df.withColumn(col_name, col(col_name).cast(col_data_type))

display(df)

# COMMAND ----------

[df.columns]

# COMMAND ----------

# Dropping unwanted columns

# List of columns to drop
columns_to_drop = ['id',
  'name',
  'host_id',
  'host_name',
  'last_review',
  ]

# Drop the columns from our dataframe
df_dropped = df.drop(*columns_to_drop)

# Verify the columns are dropped from the dataframe
df_dropped.columns

# COMMAND ----------

# Handle missing values
from pyspark.sql import functions as F

numerical_cols = ['latitude','longitude','price','minimum_nights', 'number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']
categorical_cols = ['neighbourhood_group', 'neighbourhood', 'room_type']

# Calculate the mean for numerical columns
means = df_dropped.agg(*[F.mean(column).alias(f"mean_{column}") for column in numerical_cols]).collect()
 
# Fill missing values with means
for column in numerical_cols:
    mean_value = means[0][f"mean_{column}"]
    df_dropped = df_dropped.withColumn(column, F.when(df_dropped[column].isNotNull(), df_dropped[column]).otherwise(mean_value))
 
# Calculate the mode for categorical columns
modes = df_dropped.groupBy(*categorical_cols).count().sort(F.col("count").desc()).first()
 
# Fill missing values with modes
for column in categorical_cols:
    mode_value = modes[column]
    df_dropped = df_dropped.withColumn(column, F.when(df_dropped[column].isNotNull(), df_dropped[column]).otherwise(mode_value))

display(df_dropped)

# COMMAND ----------

# Function to remove outliers using IQR
def remove_outliers_iqr(df, columns):
    for column in columns:
        # Calculate IQR
        q1 = df.approxQuantile(column, [0.25], 0.01)[0]
        q3 = df.approxQuantile(column, [0.75], 0.01)[0]
        iqr = q3 - q1

        # Define the lower and upper bounds to filter outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter out rows outside the bounds
        df = df.filter((col(column) >= lower_bound) & (col(column) <= upper_bound))

    return df

# COMMAND ----------

# Remove outliers from the numerical columns
df_filtered = remove_outliers_iqr(df_dropped, numerical_cols)

display(df_filtered)

# COMMAND ----------

# One hot encoding of categorical columns
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

stages = []
for column in categorical_cols:
    # Indexing each categorical column
    string_indexer = StringIndexer(inputCol=column, outputCol=f"{column}_indexed")
    one_hot_encoder = OneHotEncoder(inputCol=f"{column}_indexed", outputCol=f"{column}_encoded")
    
    # Adding the indexers and encoders to stages
    stages += [string_indexer, one_hot_encoder]

# Setting up the Pipeline
pipeline = Pipeline(stages=stages)

# Fitting and transforming the data
model = pipeline.fit(filtered_df)
transformed_df = model.transform(filtered_df)
display(transformed_df)

# COMMAND ----------

#Apply StandardScaler to Normalize Data
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# Combining features into a single vector column
vec_assembler = VectorAssembler(inputCols=['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365','neighbourhood_group_encoded','neighbourhood_encoded','room_type_encoded'], outputCol="features")

# Applying Standard Scaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Setting up the Pipeline
pipeline = Pipeline(stages=[vec_assembler, scaler])
model = pipeline.fit(transformed_df)
scaled_df = model.transform(transformed_df)
display(scaled_df)

# COMMAND ----------

# Train a RandomForestRegressor model
from pyspark.ml.regression import RandomForestRegressor

# Split the dataset into train and test
train_data, test_data = scaled_df.randomSplit([0.8, 0.2], seed=1234)

# Define the RandomForestRegressor model
rf_model = RandomForestRegressor(featuresCol="scaled_features", labelCol="price")

# Fit the model to the training data
model = rf_model.fit(train_data)

# Evaluate the model on test data
from pyspark.ml.evaluation import RegressionEvaluator

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using RMSE (Root Mean Squared Error)
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")

# COMMAND ----------

# MAGIC %md
# MAGIC RMSE has the unit same as the price that is dollars. It implies on average, the model's predictions are closer to the actual values. The goal is to minimize the RMSE.

# COMMAND ----------

# MAGIC %md
# MAGIC https://spark.apache.org/docs/latest/ml-classification-regression.html#regression
