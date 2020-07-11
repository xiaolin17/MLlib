import findspark
findspark.init()

from pyspark.ml.feature import Bucketizer
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

splits = [-float("inf"),-0.5,0.0,0.5,float("inf")]
data = [(-999.9,),(-0.5,),(-0.3,),(0.0,),(0.2,),(999.9,)]
dataFrame = spark.createDataFrame(data,["features"])

bucketizer = Bucketizer(splits=splits,
                        inputCol="features",
                        outputCol="bucketedFeatures")

bucketedData = bucketizer.transform(dataFrame)
bucketedData.show()

