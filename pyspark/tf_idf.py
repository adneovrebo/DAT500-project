import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF
import numpy as np
import pandas as pd
from pyspark.sql.types import *
import pyspark.sql.functions as psf
from pyspark.sql.types import StringType



def array_to_string(my_list):
    return '[' + ','.join([str(elem) for elem in my_list]) + ']'

def split_file(x):
    value=x.split('\t')
    return (value[0], value[1].split(' '))

def check_similarity(x, df):
    sim = []
    for y in df.collect():
        sim_score = np.dot(x.idf, y.idf) / (np.linalg.norm(x.idf) * np.linalg.norm(y.idf))
        sim.append(sim_score)
    return sim




if __name__ == "__main__":

    # create Spark context with Spark configuration
    # conf = SparkConf().setAppName("Word Count - Python") #.set("spark.hadoop.yarn.resourcemanager.address", "192.168.0.104:8032")
    # sc = SparkContext(conf=conf)

    spark = SparkSession.builder.appName('TF-IDF').getOrCreate()
    sc = spark.sparkContext 
    
    data = (sc.textFile('cleaned.txt')
    .map(lambda x: split_file(x))
    .toDF(['id', 'words'])
    )

    hashingTF = HashingTF(inputCol="words", outputCol='features', numFeatures=1000)
    tf = hashingTF.transform(data)

    idf = IDF(inputCol='features', outputCol='idf')
    model = idf.fit(tf)
    tf_idf = model.transform(tf)

    columns = ['id', 'sim_score_floats']

    cosine_similarity_udf = psf.udf(lambda x,y: round(float(x.dot(y)/(x.norm(2) * y.norm(2))), 4), DoubleType())
    result = tf_idf.alias("i").join(tf_idf.alias("j"), psf.col("i.id") < psf.col("j.id"))\
        .select(
            psf.col("i.id").alias("i"), 
            psf.col("j.id").alias("j"), 
            cosine_similarity_udf("i.idf", "j.idf").alias("similarity"))\
        .sort("i", "j")


    result.coalesce(1).write.csv('similarity_result')
