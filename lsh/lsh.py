from mrjob.job import MRJob
from mrjob.step import MRStep
import ast
import random

def shuffle_and_return(x):
  random.shuffle(x)
  return x

class LSH(MRJob):
    '''
        Generating signatures
    '''
    def configure_args(self):
        super(LSH, self).configure_args()
        self.add_file_arg('--vocab')
        self.add_passthru_arg("--bands", default=10)
        self.add_passthru_arg("--hash_functions", default=100)

    def steps(self):
        return [
            MRStep(
                mapper_init=self.bucket_mapper_init,
                mapper=self.bucket_mapper,
                reducer=self.bucket_reducer
            ), 
            MRStep(
                mapper=self.filter_mapper,
                reducer=self.filter_reducer
            )
        ]

    '''
        Generate feature vector for each article
    '''
    def bucket_mapper_init(self):
        # Read the vocab file and store it in a list
        with open(self.options.vocab) as f:
            self.vocab = list(ast.literal_eval(f.read().split("\t")[1]))

        random.seed(42)
        numrows = 100
        
        # Set numpy random seed
        self.signature = [shuffle_and_return(list(range(1, len(self.vocab) + 1))) for _ in range(numrows)]

        self.n_ngrams = len(self.vocab[0].split())

        # Number of bands etc...
        self.bands = int(self.options.bands)
        self.hash_functions = int(self.options.hash_functions)

        # Hash functions must be divisible by bands
        assert self.hash_functions % self.bands == 0

        self.bands_step_size = self.hash_functions // self.bands


    def bucket_mapper(self, _, line):
        # Get the article id and the article text
        article_id, article_text = line.replace('"','').split('\t')

        # Get the ngrams from the article text
        article_text_splitted = article_text.split()
        n_grams = zip(*[article_text_splitted[i:] for i in range(int(self.n_ngrams))])

        # Join all lists in n_grams into string
        n_grams_joined = set([" ".join(x) for x in n_grams])

        # Find the intersection of the n_grams_joined and the vocab
        shingled_sparse_vector = [1 if x in n_grams_joined else 0 for x in self.vocab]

        # Get the signature for the article
        hash_signature = []

        # Get the indexes where the value is 1, much smaller subset to work with
        indexes_eq_1 = [i for i, x in enumerate(shingled_sparse_vector) if x == 1]

        for hash_function in self.signature:
            # Get the hash value that corresponds to a one in the sparse vector
            matches = [hash_function[i] for i in indexes_eq_1]

            # Then we can loop over all the matches and find the minimum value
            if len(matches) == 0:
                # There is no match, so we can just set the hash value to the maximum value
                hash_signature.append(max(hash_function))
            else:
                hash_signature.append(min(matches))

        # Divide signature into bands
        hash_signature_bands = [hash_signature[i:i+self.bands_step_size] for i in range(0, len(hash_signature), self.bands_step_size)]

        # For all bands generate a hash value
        for band in hash_signature_bands:
            yield str(band), article_id

    def bucket_reducer(self, key, bucket_id):
        yield key, list(bucket_id)

    '''
        Filter the buckets that have more than one article and store article id with corresponding articleids
    '''
    def filter_mapper(self, _, article_ids):
        if len(article_ids) > 1:
            for article_id in article_ids:
                for article_id_2 in article_ids:
                    if article_id != article_id_2:
                        yield article_id_2, article_id

    def filter_reducer(self, key, values):
        yield key, list(set(values))
        
if __name__ == '__main__':
    LSH.run()