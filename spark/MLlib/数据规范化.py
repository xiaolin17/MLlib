import findspark
findspark.init()

from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.5, -1.0]),),
    (1, Vectors.dense([2.0, 1.0, 1.0]),),
    (2, Vectors.dense([4.0, 10.0, 2.0]),),
],
    ["id","features"]
)

normalizer = Normalizer(inputCol="features",
                        outputCol="normFeatures",
                        p=1.0)

l1NormData = normalizer.transform(dataFrame)
print("Normalized using L^1 norm")
l1NormData.show()

lInfNormData = normalizer.transform(dataFrame,{normalizer.p:float("inf")})
print("Normalized using L^inf norm")
lInfNormData.show()