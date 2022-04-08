from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, concat_ws, row_number, monotonically_increasing_id, count, explode
from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql.types import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from operator import attrgetter
from scipy.sparse import vstack
import numpy as np
from pyspark.ml.linalg import DenseVector
from pyspark.ml.functions import vector_to_array
from pathlib import Path
import shutil

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

def calculated_cosine_similarity(sources, targets, threshold=.8):
    cosimilarities = cosine_similarity(sources.toarray(), targets.toarray())
    for _, cosimilarity in enumerate(cosimilarities):
        cosimilarity = cosimilarity.flatten()
        rounded = [np.round(x, 4) for x in cosimilarity]
        # Find the best match by using argsort()[-1]
        # target_index = cosimilarity.argsort()[-1]
        # source_index = inputs_start_index + i
        # similarity = cosimilarity[target_index]
        # if cosimilarity >= threshold:
        yield rounded

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName('TF-IDF') \
        .config("spark.sql.analyzer.maxIterations", "500") \
        .config('spark.executor.memory', '8g') \
        .config('spark.driver.memory', '8g') \
        .getOrCreate()
    sc = spark.sparkContext 

    rdd = (sc.textFile('cleaned2.txt') #'hdfs://namenode:9000/arxiv_dataset/cleaned.txt'
        .map(lambda line: line.split('\t'))
        .map(lambda r: (r[0], r[1].split(" "))))

    schema = StructType([
            StructField('id', StringType()),
            StructField('words', ArrayType(elementType=StringType()))
    ])

    data = spark.createDataFrame(rdd, schema)

    hashingTF = HashingTF(inputCol="words", outputCol='features', numFeatures=2**14)
    tf = hashingTF.transform(data)

    idf = IDF(inputCol='features', outputCol='idf')
    model = idf.fit(tf)
    tf_idf = model.transform(tf)
    tf_idf = tf_idf.drop('words', 'features')

    vectors = tf_idf.rdd.map(attrgetter('idf'))
    matrix = vectors.map(as_matrix)
    matrix_reduced = matrix.reduce(lambda x, y: vstack([x, y]))
    matrix_parallelized = parallelize_matrix(matrix_reduced, rows_per_chunk=100)
    matrix_broadcast = broadcast_matrix(matrix_reduced)
    res = matrix_parallelized.flatMap(lambda submatrix: \
        calculated_cosine_similarity(csr_matrix(submatrix[1], \
            shape=submatrix[2]), matrix_broadcast, submatrix[0]))

    
    result = res.map(lambda x: (x[0].tolist(), DenseVector(x[0:]))).toDF()

    data = data.drop('words')


    final = result.drop('_1')
    final = final.withColumn('_2', vector_to_array('_2'))

    final = final.select("*", explode("_2").alias("exploded"))\
        .where(col("exploded") > 0.2)\
        .groupBy("_2")\
        .agg(count("exploded").alias("sim_count"))\
        .drop('_2')


    # final = final.withColumn('_2', final._2.cast(ArrayType(elementType=StringType())))
    # final = final.withColumn('_2', concat_ws(",",col("_2")))

    # data=data.withColumn('row_index', row_number().over(Window.orderBy(monotonically_increasing_id())))
    # final=final.withColumn('row_index', row_number().over(Window.orderBy(monotonically_increasing_id())))

    # final = data.join(final, on=["row_index"]).drop("row_index")

    dirpath = Path('output')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    final.write.csv('output')



