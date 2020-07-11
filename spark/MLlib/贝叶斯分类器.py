import findspark
findspark.init()

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

nb=NaiveBayes(smoothing=1.0,modelType="multinomial")
data=spark.read.format("libsvm").load("D:/spark/data/mllib/sample_libsvm_data.txt")
splits=data.randomSplit([0.6,0.4],1234)
train = splits[0]
test = splits[1]
model = nb.fit(train)
predictions = model.transform(test)
predictions.show()

evaluator = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accurary = "+str(accuracy))