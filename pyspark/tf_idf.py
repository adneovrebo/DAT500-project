from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql.types import *
import pyspark.sql.functions as psf
from pyspark.sql.types import StringType, StructType

def split_file(x):
    value=x.split('\t')
    return (value[0].replace('"', ''), value[1].split(' '))

def cosine_similarity(x, y):
    return  round(float(x.dot(y)/(x.norm(2) * y.norm(2))), 4)


if __name__ == "__main__":

    # create Spark context with Spark configuration
    # conf = SparkConf().setAppName("Word Count - Python") #.set("spark.hadoop.yarn.resourcemanager.address", "192.168.0.104:8032")
    spark = SparkSession.builder \
        .appName('TF-IDF') \
        .config("spark.sql.analyzer.maxIterations", "500") \
        .config('spark.executor.memory', '8g') \
        .config('spark.driver.memory', '8g') \
        .getOrCreate()
    sc = spark.sparkContext 

    rdd = (sc.textFile('hdfs://namenode:9000/arxiv_dataset/cleaned.txt')
        .map(lambda line: line.split('\t'))
        .map(lambda r: (r[0], r[1].split(" "))))

    schema = StructType([
            StructField('id', StringType()),
            StructField('words', ArrayType(elementType=StringType()))
    ])

    data = spark.createDataFrame(rdd, schema).limit(100)

    hashingTF = HashingTF(inputCol="words", outputCol='features', numFeatures=2**18)
    tf = hashingTF.transform(data)

    idf = IDF(inputCol='features', outputCol='idf')
    model = idf.fit(tf)
    tf_idf = model.transform(tf)
    tf_idf = tf_idf.drop('words', 'features', 'categories')

    cosine_similarity_udf = psf.udf(lambda x,y: round(float(x.dot(y)/(x.norm(2) * y.norm(2))), 4), DoubleType())

    res = tf_idf.alias("i").join(tf_idf.alias("j"), psf.col("i.id") < psf.col("j.id"))\
            .select(
                psf.col("i.id").alias("i"), 
                psf.col("j.id").alias("j"), 
                cosine_similarity_udf("i.idf", "j.idf").alias("similarity"))\
            .sort("i", "j")
    res = res.filter(res.similarity > 0.2)
    if len(res.take(1)) > 0:
        res.write.csv(f'output')

