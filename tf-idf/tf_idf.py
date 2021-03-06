from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql.types import *
import pyspark.sql.functions as psf
from pyspark.sql.types import StringType, StructType
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from operator import attrgetter
from scipy.sparse import vstack
import numpy as np

def as_matrix(vec):
    data, indices = vec.values, vec.indices
    shape = 1, vec.size
    return csr_matrix((data, indices, np.array([0, vec.values.size])), shape)

def broadcast_matrix(mat):
    bcast = sc.broadcast((mat.data, mat.indices, mat.indptr))
    (data, indices, indptr) = bcast.value
    bcast_mat = csr_matrix((data, indices, indptr), shape=mat.shape)
    return bcast_mat 

def parallelize_matrix(scipy_mat, rows_per_chunk=100):
    [rows, cols] = scipy_mat.shape
    i = 0
    submatrices = []
    while i < rows:
        current_chunk_size = min(rows_per_chunk, rows - i)
        submat = scipy_mat[i:i + current_chunk_size]
        submatrices.append((i, (submat.data, submat.indices, submat.indptr), (current_chunk_size, cols)))
        i += current_chunk_size
    return sc.parallelize(submatrices)

def calculated_cosine_similarity(sources, targets, threshold=.2):
    cosimilarities = cosine_similarity(sources.toarray(), targets.toarray())
    for cosimilarity in cosimilarities:
        res_list = []
        cosimilarity = cosimilarity.flatten()
        for score in cosimilarity:
            if score > threshold:
                res_list.append(score)
        
        yield len(res_list)-1

if __name__ == "__main__":

    spark = SparkSession.builder \
        .appName('TF-IDF') \
        .config("spark.sql.analyzer.maxIterations", "500") \
        .getOrCreate()
    sc = spark.sparkContext
    
    # Read the articles into a RDD
    rdd = (sc.textFile('cleaned-test.txt') # To run on cluster use: 'hdfs://namenode:9000/arxiv_dataset/cleaned.txt'
        .map(lambda line: line.split('\t'))
        .map(lambda r: (r[0].replace('"', ''), r[1].split(" "))))

    # Schema for DataFrame
    schema = StructType([
        StructField('id', StringType()),
        StructField('words', ArrayType(elementType=StringType()))
    ])

    # Convert RDD to DataFrame
    data = spark.createDataFrame(rdd, schema)


    # Read categories into dataframe
    categories = spark.read.csv('article_categories.csv', header=True, inferSchema=True)
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

    final_data = data.drop('words')
    for row in cats_list.collect():
        cat_df = joined_data.filter(joined_data[f'`{row.category}`'] == 1)
        if len(cat_df.take(2)) < 2: continue

        # Calculate TF-IDF
        hashingTF = HashingTF(inputCol="words", outputCol='features', numFeatures=2**18)
        tf = hashingTF.transform(cat_df)
        idf = IDF(inputCol='features', outputCol='idf')
        model = idf.fit(tf)
        tf_idf = model.transform(tf)
        tf_idf = tf_idf.drop('words', 'features', 'categories')

        # Calculate cosine similarity
        vectors = tf_idf.rdd.map(attrgetter('idf'))
        matrix = vectors.map(as_matrix)
        matrix_reduced = matrix.reduce(lambda x, y: vstack([x, y]))
        matrix_parallelized = parallelize_matrix(matrix_reduced, rows_per_chunk=100)
        matrix_broadcast = broadcast_matrix(matrix_reduced)
        res = matrix_parallelized.flatMap(lambda submatrix: \
            calculated_cosine_similarity(csr_matrix(submatrix[1], \
                shape=submatrix[2]), matrix_broadcast))
        
        # Save results to file
        if len(res.take(1)) > 0:
            final = round((res.sum() / 2)) # Divide by 2 because each article is compared two times, a->b and b->a
            print(f'Category: {row.category}; No. articles checked: {cat_df.count()}; Similar articles detected: {final}')
            spark.createDataFrame(spark.sparkContext.parallelize([final]), StringType()).coalesce(1).write.format("text").mode("overwrite").save(f'result/categories/{row.category}') # On cluster, use hdfs://namenode:9000/arxiv_dataset
            
