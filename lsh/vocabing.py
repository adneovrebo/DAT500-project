from mrjob.job import MRJob
from mrjob.step import MRStep
import	heapq

class ArXivVocaber(MRJob):
    def configure_args(self):
        super(ArXivVocaber, self).configure_args()
        self.add_passthru_arg("--ngrams", default=5)
        self.add_passthru_arg("--top_ngrams", default=1000)
    
    def steps(self):
        return [
            MRStep(
                mapper=self.ngram_mapper, 
                combiner=self.ngram_combiner, 
                reducer=self.ngram_reducer
            ),
            MRStep(
                mapper=self.top_ngrams_mapper,
                reducer=self.top_ngrams_reducer
            )
            ]

    '''
        Finding number of ngrams over all articles
    '''
    def ngram_mapper(self, _, text):
        if text:
            text_splitted = text.split()
            ngrams = zip(*[text_splitted[i:] for i in range(int(self.options.ngrams))])
            for ngram in ngrams:            
                yield " ".join(ngram), 1

    def ngram_combiner(self, ngram, count):
        yield ngram, sum(count)

    def ngram_reducer(self, ngram, count):
        yield ngram, sum(count)

    '''
        Sorting ngrams by frequency and keeping top n
    '''
    def top_ngrams_mapper(self, ngram, count):
        yield None, (count, ngram)

    def top_ngrams_reducer(self, _, ngram_count):
        for	countAndWord in heapq.nlargest(int(self.options.top_ngrams), ngram_count):
            yield _, countAndWord[1]

if __name__ == '__main__':
    ArXivVocaber.run()