from mrjob.job import MRJob
import json
import re

class PreProcessArxivArticles(MRJob):
    def mapper(self, _, line):
        article_entry = json.loads(line)

        # Join the list article_text wit eachother
        article_text = str(''.join(article_entry['article_text']))

        # Remove latex tags
        article_text = re.sub(r'\\[a-zA-Z]+', '', article_text)
        article_text = re.sub(r'@[a-zA-Z]+', '', article_text)
        
        # Remove newlines, tabs and alphanumeric characters
        article_text = re.sub(r'[^\w\s]', '', article_text)

        # Replace occurences of mulitple spaces with a single space
        article_text = re.sub(r'\s+', ' ', article_text)

        # Ensure that the article text is lowercase
        article_text = article_text.lower()

        yield article_entry['article_id'],  article_text

    def combiner(self, key, text):
        yield key, list(text)[0]

    def reducer(self, key, text):
        yield key, list(text)[0]

if __name__ == '__main__':
    PreProcessArxivArticles.run()