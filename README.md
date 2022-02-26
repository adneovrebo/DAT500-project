# DAT500 project - Group 7

- Ådne Øvrebø
- Vegard Matre


Run commands to check for similarities in the data:
```bash
# Version 1
python3 preprocessing.py < sample.txt > cleaned.txt
python3 vocabing.py < cleaned.txt > vocab.txt --top_ngrams=10000 --ngrams=3
python3 lsh.py < cleaned.txt > lsh.txt --vocab vocab.txt
```