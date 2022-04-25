# LSH 

## Runing vocabing example
Run commands to check for similarities in the dataset:
```bash
# Running locally
python3 vocabing.py < cleaned.txt > vocab.txt --top_ngrams=10000 --ngrams=1

# Running on cluster
python3 vocabing.py --top_ngrams=200000 --ngrams=8 -r hadoop hdfs:///arxiv_dataset/cleaned.txt --output-dir hdfs:///arxiv_dataset/vocabing.txt --no-output
```

## Running lsh.py example (MRJob)
```bash
# Running locally
python3 lsh.py < example.txt > lsh.txt --vocab vocab.txt  --threshold 0.2 --bands 10 --hash_functions 100

# Running on cluster
python3 lsh.py < cleaned.txt > lsh.txt --vocab  hdfs:///arxiv_dataset/vocabing.txt  --threshold 0.8 --bands 100 --hash_functions 300
```

## Running lsh.ipynb example (Spark)
```bash
# Runned using interactive session
# Start interactive session
pyspark

# Paste in code
```