import pyspark

sc = pyspark.SparkContext('local[*]')

txt = sc.textFile('lorem_ipsum.txt')
python_lines = txt.filter(lambda line: 'python' in line.lower())

with open('results.txt', 'w') as file_obj:
    file_obj.write(f'Number of lines: {txt.count()}\n')
    file_obj.write(f'Number of lines with python: {python_lines.count()}\n')

