import findspark
findspark.init()

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorIndexer
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

df = spark.createDataFrame([
    (Vectors.dense([-1.0, 0.0]),),
    (Vectors.dense([0.0, 1.0]),),
    (Vectors.dense([0.0, 2.0]),),
],["a"])

indexer = VectorIndexer(maxCategories=2,
                        inputCol="a",
                        outputCol="indexed")

model=indexer.fit(df)
print(model.transform(df).head().indexed)
print(model.numFeatures)
print(model.categoryMaps)

print(indexer.setParams(outputCol="test").fit(df).transform(df).collect()[1].test)
