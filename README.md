# ArXiv plagiarism detection 🎓
[DAT500](https://www.uis.no/en/course/DAT500_1) - Data-intensive Systems - University of Stavanger
### Authors ✍🏻
- [Ådne Øvrebø](https://github.com/adneovrebo)
- [Vegard Matre](https://github.com/vmatre)

## Project abstraction
In similarity detection, TF-IDF ( Term Frequency - Inverse Document Frequency ) and LSH (Locality Sensitive Hashing) are two widely used algorithms with different advantages. This paper focuses on implementing the two algorithms using the Hadoop and Spark ecosystem and preprocessing the data to be used with the algorithms. The mentioned algorithms will be used to find similar documents that might be plagiarism in the arXiv dataset, an open scholarly article dataset. LSH is implemented using both the MapReduce framework MRJob and Spark with MLLib. Implementation using MRJob is harder but allows for a deeper understanding of the workings behind the more abstract frameworks such as Spark. TF-IDF is implemented using Spark and MLLib. The mentioned frameworks allow for processing a large amount of data distributed over multiple machines in a cluster. 

## Technology stack 
- [Hadoop](https://hadoop.apache.org/)
- [MRJob](https://mrjob.readthedocs.io/en/latest/)
- [Spark](https://spark.apache.org/)
- [PySpark](https://spark.apache.org/docs/latest/api/python/pyspark.html)
    - [MLLib](https://spark.apache.org/docs/latest/mllib-guide.html)

## Datasets
- [Tensorflow arXiv dataset](https://www.tensorflow.org/datasets/catalog/scientific_papers)
- [arXiv metadata](https://www.kaggle.com/datasets/Cornell-University/arxiv)

⚠️ NB: Dataset is not included in this repository.

## Repo structure
- preprocessing 
    - [preprocessing.py](preprocessing/preprocessing.py)
        - MRJob implementation
- lsh
    - [vocabing.py](lsh/vocabing.py)
        - MRJob for finding vocabulary
    - [lsh.py](lsh/lsh.py)
        - MRJob implementation
    - [lsh.ipynb](lsh/lsh.ipynb)
        - Spark implementation
    - [plotting.ipynb](lsh/plotting.ipynb)
        - Plotting probability of documents being in the same bucket.
- tf-idf
    - [tf_idf.py](tf-idf/tf_idf.py)
        - Spark implementation
