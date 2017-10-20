from collections import defaultdict
from math import log, exp
import numpy
from count_dict import gt_prob, CustomDict



class ProbabilityDict(object):
    def __init__(self, count_dict):
        self.counts = count_dict
        self.vocab_size_total = sum(self.counts.unigrams.values())
        self.vocab_size_unique = len(self.counts.unigrams.keys())

        
        
    # Maximum Likelihood Estimate
        
    def unigram_MLE(self):
        mle = defaultdict(float)
        for key, value in self.counts.unigrams.items():
            mle[key] = value/self.vocab_size_total
            
        return mle

    
    def bigram_MLE(self):
        mle = defaultdict(float)
        for key, value in self.counts.bigrams.items():
            w1, w2 = key
            mle[key] = value/self.counts.unigrams[(w1,)]

        return mle

    
    # def trigram_MLE(self):
    #     mle = defaultdict(float)
    #     for key, value in self.counts.trigrams.items():
    #         w1, w2, w3 = key
    #         mle[key] = value/self.counts.bigrams[(w1, w2)]

    #     return mle

    # Laplace Smoothed probabilities
    
    def unigram_laplace(self):
        if_missing = lambda x: (1./(self.vocab_size_total + self.vocab_size_unique + 1))
        laplace_dict = CustomDict(if_missing)

        for key, value in self.counts.unigrams.items():
            laplace_dict[key] = (value + 1)/(self.vocab_size_total + self.vocab_size_unique + 1)

        return laplace_dict


    def bigram_laplace(self):
        bigram_profile = defaultdict(set)
        
        """
        Bigram profile keys should be unigrams such that 
        profile[the] -> [(the dog), (the cat), (the monkey)]
        so that you can access the number of words w such that c(w|the) > 1
        """
        for key in self.counts.bigrams:
            w1, w2 = key
            bigram_profile[w1].add(w2)

        # if_missing = lambda x: (1.0/(self.counts.unigrams[(x[0],)] + len(bigram_profile[x[0]]) + 1))
        if_missing = lambda x: (1/(self.counts.unigrams[(x[0],)] + self.vocab_size_unique + 1))
        laplace_dict = CustomDict(if_missing)
            
        for key, value in self.counts.bigrams.items():
            w1, w2 = key
            laplace_dict[key] = (value + 1)/(self.counts.unigrams[(w1,)] + self.vocab_size_unique + 1)
        return laplace_dict

    
    # def trigram_laplace(self):
    #     bigram_vocab_dict = defaultdict(lambda: defaultdict(int))
    #     for trigram in self.counts.trigrams.keys():
    #         w1, w2, w3 = trigram
    #         bigram_vocab_dict[(w1, w2)][(w1, w2, w3)] += 1
        
    #     def if_missing(x):
    #         w1, w2, w3 = x
    #         bigram_vocab_size = len(bigram_vocab_dict[(w1, w2)].keys())
    #         return (1.0/(self.counts.bigrams[(w1, w2)] + bigram_vocab_size + 1))
        
    #     laplace_dict = CustomDict(if_missing)

    #     for key, value in self.counts.trigrams.items():
    #         w1, w2, w3 = key
    #         bigram_vocab_size = len(bigram_vocab_dict[(w1, w2)].keys())
    #         laplace_dict[key] = (value + 1)/(self.counts.bigrams[(w1, w2)] + bigram_vocab_size + 1)
            
    #     return laplace_dict


    def unigram_gt_MLE(self):
        gt = self.counts.gt_unigrams
        gt_dict = CustomDict(lambda key : gt[key]/self.vocab_size_total)
        gt_dict[("the",)] = gt_dict[("the", )]
        return gt_dict

    def bigram_gt_MLE(self):
        gt = self.counts.gt_bigrams
        gtu = self.counts.gt_unigrams
        gt_dict = CustomDict(lambda key : gt[key]/gtu[(key[0],)])
        gt_dict[("<s>", "the")] =  gt_dict[("<s>", "the")]
        return gt_dict
    
    # def trigram_gt_MLE(self):
    #     gt = self.counts.gt_trigrams
    #     gtb = self.counts.gt_bigrams
    #     def test(key):
    #         val = gt[key]/gtb[(key[0], key[1])]
    #         if val >= 1:
    #             print(key, val)
    #         return val
    #     gt_dict = CustomDict(lambda key: gt[key]/gtb[(key[0], key[1])])
    #     gt_dict = CustomDict(test)
    #     gt_dict[("<s>", "the", "company")] = gt_dict[("<s>", "the", "company")]
    #     return gt_dict
    
    def unigram_gt_laplace(self):
        gt = self.counts.gt_unigrams
        gt_dict = CustomDict(lambda key : (gt[key] + 1)/(self.vocab_size_total + self.vocab_size_unique + 1))
        gt_dict[("the",)] = gt_dict[("the", )]
        return gt_dict

    def bigram_gt_laplace(self):
        gt = self.counts.gt_bigrams
        gtu = self.counts.gt_unigrams

        bigram_profile = defaultdict(set)
        
        """
        Bigram profile keys should be unigrams such that 
        profile[the] -> [(the dog), (the cat), (the monkey)]
        so that you can access the number of words w such that c(w|the) > 1
        """
        for key in self.counts.bigrams:
            w1, w2 = key
            bigram_profile[w1].add(key)

        def test(key):
            val = (gt[key] + 1)/(gtu[(key[0],)] + self.vocab_size_unique + 1)
            return val 

        gt_dict = CustomDict(test)
        gt_dict[("<s>", "the")] =  gt_dict[("<s>", "the")]
        return gt_dict

        
    # def trigram_gt_laplace(self):
    #     bigrams_vocab_size = len(self.counts.bigrams.keys())
        
    #     def if_missing(x):
    #         return ((gt_prob(1, self.counts.gt_trigram_counts) + 1)/(self.counts.bigrams[x[0], x[1]] + bigrams_vocab_size + 1))

    #     gt_dict = CustomDict(if_missing)
        
    #     for key, value in self.counts.trigrams.items():
    #         w1, w2, w3 = key
    #         c_star_trigram = (value + 1) * (gt_prob(value + 1, self.counts.gt_trigram_counts)/gt_prob(value, self.counts.gt_trigram_counts))
    #         gt_dict[key] = (c_star_trigram + 1)/(self.counts.bigrams[(w1,w2)] + bigrams_vocab_size + 1)
            
    #     return gt_dict            
 
    # def interpolated_trigram(self):
    #     pass
    
    # Interpolated estimates 

    def interpolated_bigram(self, lambda_value):
        # Using GT MLE probabilities
        gt_bigrams = self.bigram_gt_MLE()
        gt_unigrams = self.unigram_gt_MLE()

        if_missing = lambda x: lambda_value * gt_bigrams[x] + (1 - lambda_value) * gt_unigrams[(x[1],)]
        i_dict = CustomDict(if_missing)

        for key, value in gt_bigrams.items():
            i_dict[key] = (lambda_value * value) + ((1 - lambda_value) * gt_unigrams[(key[1],)])
            
        return i_dict
        

    def stupid_backoff_bigram(self):
        mle_bigrams = self.bigram_MLE()
        gt_unigrams = self.unigram_gt_MLE()

        def if_missing(key):
            w1, w2 = key
            return gt_unigrams[(w2,)] * .4
        
        sb_dict = CustomDict(if_missing)
        for key, value in mle_bigrams.items():
            sb_dict[key] = value

        return sb_dict
        
    def kn_bigram(self):
        pass

    def katz_bigram(self):
        pass
