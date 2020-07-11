import findspark
findspark.init()

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer,VectorAssembler
sc = SparkContext('local')
spark = SparkSession(sc)

featureCols = [
    "balance","duration","history","purpose","amount",
    "saving","employment","instPercent","sexMarried","guarantors",
    "residenceDuration","assets","age","concCredit","apartment",
    "credits","occupation","dependents","hasPhone","foreigh"
]

df = spark.read.csv("C:/Users/xiaohuaidan/Desktop/新建文件夹1/germancredit.csv",header=True,inferSchema=True)
assembler = VectorAssembler().setInputCols([e for e in df.columns if e not in featureCols]).setOutputCol("features")
df2 = assembler.transform(df)
labelIndexer = StringIndexer().setInputCol("creditability").setOutputCol("label")
df3 = labelIndexer.fit(df2).transform(df2)
(trainingData, testData) = df3.randomSplit([0.7,0.3],5043)
dt = DecisionTreeClassifier(labelCol="creditability",featuresCol="features")
model = dt.fit(trainingData)
predictions = model.transform(testData)
predictions.select("prediction","probability","rawPrediction").show(10,False)
