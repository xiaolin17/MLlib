import os
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import findspark
findspark.init()

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import Tokenizer,HashingTF,IDF,StringIndexer
from pyspark.sql.types import *
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

def gat_dataset(sc,dataset):
    total_rdd = sc.parallelize([])
    for dir_name in os.listdir(dataset):
        dir_path = os.path.join(dataset,dir_name)
        f_rdd = sc.wholeTextFiles(dir_path)
        f_rdd2 = f_rdd.map(lambda x:add_category(dir_name,x))
        total_rdd = total_rdd.union(f_rdd2)

    schema = StructType([
        StructField("catagory",StringType(),True),
        StructField("file path", StringType(), True),
        StructField("content", StringType(), True)
    ])

    f_df = spark.createDataFrame(total_rdd,schema)
    return f_df

dataset = gat_dataset(sc,'20news-19997/20_newsgroups')
tokenzer = Tokenizer(inputCol="content",outputCol="words")
wordsData = tokenzer.transform(dataset)
hashingTF = HashingTF(inputCol="words",outputCol="ravFeatures",numFeatures=500)
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures",outputCol="features")
idfModel = idf.fit(featurizedData)
df = idfModel.transform(featurizedData)

labelIndexer = StringIndexer().setInputCol("category").setOutputCol("label")
labelIndexer_model = labelIndexer.fit(df)
df2 = labelIndexer_model.transform(df)
df2.select('category','label').show(5,truncate = False)

(trainingData, testData) = df2.randomSplit([0.7,0.3], 2134)
nb = NaiveBayes(smoothing=1.0,modelType="multinomial")
model = nb.fit(trainingData)
prediction = model.transform(testData)
prediction.select('category','label','probability','prediction').show()