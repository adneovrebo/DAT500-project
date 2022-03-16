import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF
import numpy as np
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType



def array_to_string(my_list):
    return '[' + ','.join([str(elem) for elem in my_list]) + ']'

def split_file(x):
    value=x.value.split('\t')
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


    spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
    df = spark.read.text('cleaned.txt')
    rdd = df.rdd.map(lambda x: split_file(x))
    df2 = rdd.toDF().withColumnRenamed('_2', 'content').withColumnRenamed('_1', 'id')

    hashingTF = HashingTF(inputCol="content", outputCol='features')
    hashingTF.setNumFeatures(1000)

    tf = hashingTF.transform(df2)

    idf = IDF()
    idf.setInputCol('features')
    idf.setOutputCol('idf')
    model = idf.fit(tf)
    tf_idf = model.transform(tf)

    columns = ['id', 'sim_score_floats']

    data = []
    for i, x in enumerate(tf_idf.collect()):
        sim_score_list = []
        for y in tf_idf.collect():
            sim_score = x.idf.dot(y.idf) / (x.idf.norm(2) * y.idf.norm(2))
            sim_score_list.append(round(float(sim_score), 4))
        data.append((x.id, sim_score_list))

    sim_df = spark.createDataFrame(data).toDF(*columns)
    final = tf_idf.join(sim_df,["id"])

    array_to_string_udf = udf(array_to_string, StringType())

    sim_df = sim_df.withColumn('sim_score', array_to_string_udf(sim_df['sim_score_floats']))

    sim_df.drop('sim_score_floats').coalesce(1).write.csv('similarity_result')
