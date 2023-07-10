# Databricks notebook source
# MAGIC %md
# MAGIC ##homeless-rate-clustering
# MAGIC ###July 8, 2023
# MAGIC __Matt Beattie, Triet Than  (University of Oklahoma)__
# MAGIC
# MAGIC This script takes the HUD data and clusters combinations of state and year using per capita sheltered and unsheltered homeless rates.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, IntegerType, FloatType

import pandas as pd
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("App").getOrCreate()



# COMMAND ----------

# DBTITLE 1,Read in data from ACS and HUD
# Create the clustering input data from ACS and HUD data
df1 = spark.sql("""
  SELECT 
    a.state,
    a.year,
    a.hmls_unsh AS hmls_unshelt,
    (a.hmls_es + a.hmls_th + a.hmls_sh) AS hmls_shelt,
    b.pop_tot,
    a.hmls_unsh/b.pop_tot AS hmls_unshelt_pcp,
    (a.hmls_es + a.hmls_th + a.hmls_sh)/b.pop_tot AS hmls_shelt_pcp
  FROM homelessness.hud_pit_and_hic a
  JOIN homelessness.acs_data_features b
  ON a.state = b.statecode
  AND a.year = b.year
""")

display(df1)

# COMMAND ----------

# DBTITLE 1,Generate silhouette data to plot elbow method
# Prepare data for clustering
assembler = VectorAssembler(inputCols=['hmls_unshelt_pcp', 'hmls_shelt_pcp'], outputCol="features")
data = assembler.transform(df1)

# Set evaluator
evaluator = ClusteringEvaluator()

# Iterate through potential numbers of clusters
clustplotdata = []
for clustcount in range(2,10):
  # Train the model
  kmeans = KMeans(k=clustcount, seed=1)
  model = kmeans.fit(data)

  # Predict clusters for the data
  predictions = model.transform(data)
  silhouette = evaluator.evaluate(predictions)
  print("Number of clusters:", clustcount)
  print("Silhouette with squared euclidean distance = ", str(silhouette), "\n\n")
  clustplotdata.append((clustcount, silhouette))

# Put the silhouette scores into a pyspark dataframe for plotting
plotschema = StructType([StructField('clustcount', IntegerType(), True),
                     StructField('silhouette', FloatType(), True)])
clustplotdatadf = spark.createDataFrame(data=clustplotdata, schema=plotschema)

display(clustplotdatadf)


# COMMAND ----------

# DBTITLE 1,Plot the silhouette scores
silhouettepdf = clustplotdatadf.toPandas()

fig, ax = plt.subplots()
ax.set_title("Silhouette Scores of Homelessness Cluster Data")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Silhouette Score (Squared Euclidean)")
ax.plot(silhouettepdf['clustcount'], silhouettepdf['silhouette'])

plt.show()


# COMMAND ----------

# DBTITLE 1,Generate a dataset from the best number of clusters
# Train the model
kmeans = KMeans(k=4, seed=1)
model = kmeans.fit(data)

# Predict clusters for the data
predictions = model.transform(data)

# Show the results and load into a view
predictions.createOrReplaceTempView('predictionstbl')
display(predictions)


# COMMAND ----------

# DBTITLE 1,Count the items in the clusters
# MAGIC %sql
# MAGIC SELECT prediction, count(*) AS cnt FROM predictionstbl
# MAGIC GROUP BY prediction

# COMMAND ----------

# DBTITLE 1,Visualize the clusters
pdf = predictions.toPandas()

fig, ax = plt.subplots()
ax.set_title("Clusters in Unsheltered vs Sheltered Homelessness")
ax.set_xlabel("Sheltered Homelessness (per capita)")
ax.set_ylabel("Unsheltered Homelessness (per capita)")
scatter = ax.scatter(pdf['hmls_shelt_pcp'], pdf['hmls_unshelt_pcp'], c=pdf['prediction'],
           label=pdf['prediction'])

legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Cluster")
ax.add_artist(legend1)

plt.show()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT state, prediction, count(*) AS cnt FROM predictionstbl
# MAGIC GROUP BY state, prediction
# MAGIC ORDER BY state, cnt DESC

# COMMAND ----------

# DBTITLE 1,Put the clusters into a file to be used for further analysis
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE homelessness.four_clusters AS
# MAGIC SELECT
# MAGIC   state,
# MAGIC   year,
# MAGIC   prediction
# MAGIC FROM predictionstbl

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from homelessness.four_clusters
