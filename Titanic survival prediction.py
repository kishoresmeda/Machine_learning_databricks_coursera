# Databricks notebook source
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.getOrCreate()

# Load the dataset
df = spark.read.csv("dbfs:/FileStore/shared_uploads/courserabricks@outlook.com/titanic_original.csv", header=True, inferSchema=True)

# Show the dataframe
display(df)

# COMMAND ----------

# Check for missing values
from pyspark.sql.functions import col, count, isnan, when

# Rename column name home.dest to destination
df = df.withColumnRenamed("home.dest","destination")

# Check for missing values
missing_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])

display(missing_counts)

# COMMAND ----------

# Dropping the column - body because it's an identfication number only
df = df.drop("body")

# Drop rows with missing values for other columns
df = df.na.drop(subset=["pclass", "survived", "name", "sex", "sibsp", "parch", "ticket", "fare", "embarked"])

# Compute median age
median_age = df.approxQuantile("age", [0.5], 0.25)[0]

# Fill missing values
df = df.na.fill({'age': median_age, 'cabin': 'Unknown', 'boat': 'Unknown', 'destination': 'Unknown'})

# COMMAND ----------

# Handling outliers
import matplotlib.pyplot as plt

# Convert Spark DataFrame columns to Pandas Series
pandas_age = df.select("age").toPandas()
pandas_fare = df.select("fare").toPandas()

# Create box plots
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.boxplot(pandas_age["age"])
plt.title('Box plot for Age')

plt.subplot(1, 2, 2)
plt.boxplot(pandas_fare["fare"])
plt.title('Box plot for Fare')

plt.show()


# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
# Capping the 'fare' feature to the 95th percentile
# Calculate the 95th percentile for 'fare'
fare_95th_percentile = df.approxQuantile("fare", [0.95], 0.0)[0]

# Function to cap fare
def cap_fare(fare):
    return min(fare, fare_95th_percentile)

# UDF for capping fare
cap_fare_udf = udf(cap_fare, FloatType())

# Apply the UDF to cap 'fare'
df = df.withColumn("fare", cap_fare_udf("fare"))

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# Convert categorical features to numerical using StringIndexer and OneHotEncoder
categorical_cols = ['sex', 'embarked']

indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]

encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_encoded") for col in categorical_cols]

stages = indexers + encoders

# Setting up the Pipeline
pipeline = Pipeline(stages=stages)

# Fitting and transforming the data
model = pipeline.fit(df)
transformed_df = model.transform(df)
display(transformed_df)

# COMMAND ----------

# Normalize the features
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_encoded', 'embarked_encoded'],
                            outputCol='features')
# Standardize features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Setting up the Pipeline
pipeline = Pipeline(stages=[vec_assembler, scaler])
model = pipeline.fit(transformed_df)
scaled_df = model.transform(transformed_df)
display(scaled_df)

# COMMAND ----------

#Building the Machine Learning model

# Import libraries
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Create a logistic regression model
lr = LogisticRegression(featuresCol="scaled_features", labelCol="survived")

# Split the dataset into training and testing sets
train_data, test_data = scaled_df.randomSplit([0.8, 0.2], seed=42)

# Train the logistic regression model
model = lr.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)
display(predictions)

# COMMAND ----------

# Evaluate the model using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="survived")

# Calculate the area under the ROC curve
auc = evaluator.evaluate(predictions)
print(f"Area Under ROC Curve (AUC): {auc}")

# Initialize evaluators
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="survived", metricName="accuracy")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="survived", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="survived", metricName="weightedRecall")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="survived", metricName="f1")

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


