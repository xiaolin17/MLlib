import findspark
findspark.init()

from pyspark.ml.feature import StringIndexer,VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

df = spark.read.csv("C:\Users\xiaohuaidan\Desktop\新建文件夹1\iris.csv",header=True,inferSchema=True)

featureCols =["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"]

assembler = VectorAssembler().setInputCols([e for e in df.columns if e not in featureCols]).setOutputCol("features")
df2 = assembler.transform(df)
labelIndexer = StringIndexer().setInputCol("Species").setOutputCol("label")
df3 = labelIndexer.fit(df2).transform(df2)
(trainingData, testData) = df3.randomSplit([0.7,0.3],5043)
dt = RandomForestClassifier(labelCol="label",featuresCol="features",numTrees=100,maxDepth=4,maxBins=32)
model = dt.fit(trainingData)
predictions = model.transform(testData)
predictions.show(n=2,truncate=30)

