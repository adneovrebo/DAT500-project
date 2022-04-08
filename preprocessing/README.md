# Preprocessing


## Running
```bash
# Run locally example
python3 preprocessing.py < train.txt > cleaned.txt

# Run on cluster example
python3 preprocessing.py -r hadoop hdfs:///arxiv_dataset/train.txt --output-dir hdfs:///arxiv_dataset/cleaned.txt --no-output
```