import findspark
findspark.init()

from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

df = spark.createDataFrame([
    (7, Vectors.dense([0.0, 0.0, 18.0, 1.0]), 1.0,),
    (8, Vectors.dense([0.0, 1.0, 12.0, 0.0]), 0.0,),
    (9, Vectors.dense([1.0, 0.0, 15.0, 0.1]), 0.0,),
],
    ["id","features","clicked"]
)

selector = ChiSqSelector(numTopFeatures=1,
                         featuresCol="features",
                         outputCol="selected",
                         labelCol="clicked")

result = selector.fit(df).transform(df)

result.show()