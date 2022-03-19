import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import HashingTF, IDF
import numpy as np
import pandas as pd
from pyspark.sql.types import *
import pyspark.sql.functions as psf
from pyspark.sql.types import StringType
from functools import reduce
from pyspark.conf import SparkConf

def split_file(x):
    value=x.split('\t')
    return (value[0].replace('"', ''), value[1].split(' '))

def cosine_similarity(x, y):
    return  round(float(x.dot(y)/(x.norm(2) * y.norm(2))), 4)


if __name__ == "__main__":

    # create Spark context with Spark configuration
    # conf = SparkConf().setAppName("Word Count - Python") #.set("spark.hadoop.yarn.resourcemanager.address", "192.168.0.104:8032")
    spark = SparkSession.builder.appName('TF-IDF').getOrCreate()
    sc = spark.sparkContext 
    
    data = (sc.textFile('cleaned-test.txt')
    .map(lambda x: split_file(x))

    .toDF(['id', 'words'])
    )
    # Read categories into dataframe
    categories = spark.read.csv('categories.csv', header=True, inferSchema=True)
    # Make a column for each unique category, split each category on spaces
    unique_cats = categories.select('categories').distinct()
    # Split unique categories on spaces
    cats_split = unique_cats.select(psf.split(psf.col('categories'), ' ').alias('categories'))
    # Make a list of all the categories
    cats_list = cats_split.select(psf.explode(psf.col('categories')).alias('category')).distinct()
    
    # Add to categories dataframe a column for each unique category
    for row in cats_list.collect():
        category = row.category
        categories = categories.withColumn(category, psf.when(categories['categories'].contains(category), 1).otherwise(0))

    joined_data = data.join(categories, ["id"])
    hashingTF = HashingTF(inputCol="words", outputCol='features', numFeatures=1000)
    tf = hashingTF.transform(joined_data)
    # tf.show()

    idf = IDF(inputCol='features', outputCol='idf')
    model = idf.fit(tf)
    tf_idf = model.transform(tf)
    tf_idf = tf_idf.drop('words', 'features', 'categories')
    # print(tf_idf)

    
    cosine_similarity_udf = psf.udf(lambda x,y: cosine_similarity(x,y), DoubleType())
    result_append = []

    for row in cats_list.collect(): # Uses collect here since it is a limited number of categories
        cat_df = tf_idf.filter(tf_idf[f'`{row.category}`'] == 1)
        cat_df = cat_df[[cat_df.id, cat_df.idf]]

        if len(cat_df.take(1)) == 0: continue

        res = cat_df.alias("i").join(cat_df.alias("j"), psf.col("i.id") < psf.col("j.id"))\
            .select(
                psf.col("i.id").alias("i"), 
                psf.col("j.id").alias("j"), 
                cosine_similarity_udf("i.idf", "j.idf").alias("similarity"))\
            .sort("i", "j")
        result_append.append(res)

    df_series = reduce(DataFrame.unionAll, result_append).distinct()

    df_series.coalesce(1).write.csv('similarity_result')
