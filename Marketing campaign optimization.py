# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

df = spark.read.format("csv").option("header","true").option("delimiter",";").option("inferSchema","true").load("dbfs:/FileStore/shared_uploads/courserabricks@outlook.com/marketing_campaign.csv")

# COMMAND ----------

display(df)

# COMMAND ----------

display(df.summary())

# COMMAND ----------

display(df.describe())

# COMMAND ----------

# List of columns to drop
columns_to_drop = ['ID','Z_CostContact','Z_Revenue']

# Drop the columns from our dataframe
df_dropped = df.drop(*columns_to_drop)

# Verify the columns are dropped from the dataframe
df_dropped.columns

# COMMAND ----------

# Count NULL values in each column
from pyspark.sql.functions import col, sum

missing_counts = df_dropped.select([sum(col(c).isNull().cast("int")).alias(c) for c in df_dropped.columns])

# Display the missing value counts
display(missing_counts)

# COMMAND ----------

# Display the rows with missing values
target_column = 'income'
null_rows = df_dropped.filter(col(target_column).isNull())

display(null_rows)

# COMMAND ----------

import numpy as np
missing_value = np.nan  

# Drop rows where the specific column has missing values
filtered_data = df_dropped.filter(col(target_column) != missing_value)
display(filtered_data)

# COMMAND ----------

# Handling outliers
# Convert spark datafranme to pandas dataframe for visualisation purpose

import pandas as pd
df = pd.read_csv("/dbfs/FileStore/shared_uploads/courserabricks@outlook.com/marketing_campaign.csv", delimiter=';')

# Select only the numerical features
numerical_features = df.select_dtypes(include=['number']).columns.tolist()

# remove the unwanted columns
features_to_remove = ['ID', 'Z_CostContact', 'Z_Revenue']

selected_numerical_features = [x for x in numerical_features if x not in features_to_remove]

# Check the final features to visualize
selected_numerical_features

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting individual box plots for each selected numerical feature
for feature in selected_numerical_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, y=feature)
    plt.title(f'Box Plot of {feature}')
    plt.show()


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

# List of features to apply IQR on
feature_columns = ['Year_Birth',
 'Income',
 'Kidhome',
 'Teenhome',
 'Recency',
 'MntWines',
 'MntFruits',
 'MntMeatProducts',
 'MntFishProducts',
 'MntSweetProducts',
 'MntGoldProds',
 'NumDealsPurchases',
 'NumWebPurchases',
 'NumCatalogPurchases',
 'NumStorePurchases',
 'NumWebVisitsMonth']

# Apply the function to remove outliers
filtered_df = remove_outliers_iqr(filtered_data, feature_columns)

# Show the resulting DataFrame
display(filtered_df)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

categorical_columns = ['Education','Marital_Status']

stages = []
for column in categorical_columns:
    # Indexing each categorical column
    string_indexer = StringIndexer(inputCol=column, outputCol=f"{column}_indexed")
    one_hot_encoder = OneHotEncoder(inputCol=f"{column}_indexed", outputCol=f"{column}_encoded")
    
    # Adding the indexers and encoders to stages
    stages += [string_indexer, one_hot_encoder]

# Setting up the Pipeline
pipeline = Pipeline(stages=stages)

# Fitting and transforming the data
model = pipeline.fit(filtered_data)
transformed_df = model.transform(filtered_data)
display(transformed_df)

# COMMAND ----------

#Apply StandardScaler to Normalize Data
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# Combining features into a single vector column
vec_assembler = VectorAssembler(inputCols=['Year_Birth',
 'Income',
 'Kidhome',
 'Teenhome',
 'Recency',
 'MntWines',
 'MntFruits',
 'MntMeatProducts',
 'MntFishProducts',
 'MntSweetProducts',
 'MntGoldProds',
 'NumDealsPurchases',
 'NumWebPurchases',
 'NumCatalogPurchases',
 'NumStorePurchases',
 'NumWebVisitsMonth',
 'Complain',
 'Education_indexed',
 'Marital_Status_indexed'], outputCol="features")

# Applying Standard Scaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Setting up the Pipeline
pipeline = Pipeline(stages=[vec_assembler, scaler])
model = pipeline.fit(transformed_df)
scaled_df = model.transform(transformed_df)
display(scaled_df)

# COMMAND ----------

# SparkML Library docs - https://spark.apache.org/docs/latest/ml-classification-regression.html

# Importing libraries
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Split the dataset into train and test
train_data, test_data = scaled_df.randomSplit([0.8, 0.2], seed=1234)

# Training a Logistic regression problem
lr = LogisticRegression(featuresCol="scaled_features", labelCol="Response")
lr_model = lr.fit(train_data)

# Inference on the test data
predictions = lr_model.transform(test_data)

display(predictions)

# COMMAND ----------

# Evalute model performance
roc_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="Response", metricName="areaUnderROC")
roc_auc = roc_evaluator.evaluate(predictions)
print(f"Area Under ROC: {roc_auc}")

# Initialize evaluators
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="accuracy")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="weightedRecall")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="Response", metricName="f1")

# Evaluate the model
accuracy = accuracy_evaluator.evaluate(predictions)
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)
f1 = f1_evaluator.evaluate(predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# COMMAND ----------


