import imp
from mrjob.job import MRJob
from mrjob.step import MRStep
import ast
from nltk import ngrams
import numpy as np

#TODO: Få inn innstillinger skikkelig
#TODO: Check for jaccard similarity
# TODO: Improve, make it more efficient, don't use so many map reduce steps.

class LSH(MRJob):
    def steps(self):
        return [
            MRStep(
                mapper_init=self.mapper_init_feature_vector,
                mapper=self.mapper_feature_vector,
                reducer=self.reducer_feature_vector
            ), 
            MRStep(
                mapper_init=self.mapper_init_signing,
                mapper=self.mapper_signing,
                reducer=self.reducer_signing
            ),
            MRStep(
                mapper=self.filter_mapper,
                reducer=self.filter_reducer
            )
        ]

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

    '''
        Generating signatures
    '''
    def configure_args(self):
        super(LSH, self).configure_args()
        self.add_file_arg('--vocab')

    def mapper_init_signing(self):
        np.random.seed(42)
        numrows = 100
        
        # Set numpy random seed
        self.signature = np.array([np.random.permutation(np.arange(1, 10000 + 1)).tolist() for _ in range(numrows)])
    
    def mapper_signing(self, article_id, shingled_sparse_vector):
        # Get the signature for the article

        hash_signature = []

        # Get the indexes where the value is 1, much smaller subset to work with
        indexes_eq_1 = [i for i, x in enumerate(shingled_sparse_vector[0]) if x == 1]

        for hash_function in self.signature:
            # Get the hash value that corresponds to a one in the sparse vector
            matches = hash_function[indexes_eq_1]
            # Then we can loop over all the matches and find the minimum value
            if len(matches) == 0:
                # There is no match, so we can just set the hash value to the maximum value
                hash_signature.append(np.max(hash_function))
            else:
                hash_signature.append(min(matches))

        # Divide signature into bands
        hash_signature_bands = [hash_signature[i:i+10] for i in range(0, len(hash_signature), 10)]

        # For all bands generate a hash value
        for band in hash_signature_bands:
            yield hash(str(band)), article_id
    
    def reducer_signing(self, bucket_id, article_id):
        yield bucket_id, list(article_id)

    '''
        Generate feature vector for each article
    '''
    def mapper_init_feature_vector(self):
        # Read the vocab file and store it in a list
        with open(self.options.vocab) as f:
            self.vocab = list(ast.literal_eval(f.read().split("\t")[1]))


    def mapper_feature_vector(self, _, line):
        # Get the article id and the article text
        article_id, article_text = line.replace('"','').split('\t')

        # Get the ngrams from the article text
        article_text_splitted = article_text.split()
        n_grams = list(ngrams(article_text_splitted, 3))

        # Join all lists in n_grams into string
        n_grams_joined = set([" ".join(x) for x in n_grams])

        # Find the intersection of the n_grams_joined and the vocab
        shingled_sparse_vector = [1 if x in n_grams_joined else 0 for x in self.vocab]

        yield article_id, shingled_sparse_vector

    def reducer_feature_vector(self, key, shingled_sparse_vector):
        yield key, list(shingled_sparse_vector)
        
if __name__ == '__main__':
    LSH.run()