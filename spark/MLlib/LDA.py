import findspark
findspark.init()

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.clustering import LDA
from pyspark.sql.types import *
import pyspark.ml.feature as ft
sc = SparkContext('local')
spark = SparkSession(sc)

schema = StructType([StructField('documents',StringType(),True)])
text_1 = spark.read.format('text').schema(schema).load('20news-19997/20_newsgroups/alt.atheism/49960.txt')
text_2 = spark.read.format('text').schema(schema).load('20news-19997/20_newsgroups/alt.atheism/51060.txt')

text_data = text_1.union(text_2)

tokenizer =ft.RegexTokenizer(inputCol='documents',outputCol='input_arr',pattern=r'\s+|[,.\"]')
df1 = tokenizer.transform(text_data)

stopwords = ft.StopWordsRemover(inputCol='input_arr',outputCol='input_stop')
df2 = stopwords.transform(df1)

stringIndex = ft.CountVectorizer(inputCol='input_stop',outputCol='input_indexed')
cv_model = stringIndex.fit(df2)

df3 = cv_model.transform(df2)
df3.select('input_stop','input_indexed').show(truncate=False)


lda = LDA(k=2,maxIter=10,optimizer='em',featuresCol='input_indexed')
model = lda.fit(df3)
print("vocal size",model.vocabSize())
print(model.topicsMatrix)

topics = model.describeTopics()
print("The topics described by their top-weighted terms:")
topics.show(truncate=False)

result = model.transform(df3)
result.select('documents','topicDistribution').show(truncate=False)
