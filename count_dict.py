from collections import defaultdict, Counter
from math import exp,log

class CountDict(object):
    def __init__(self, filename):
        self.filename = filename
        self.GT_UPPER_BOUND = 5
        
        self.unigrams = defaultdict(int)
        self.bigrams = defaultdict(int)
        self.trigrams = defaultdict(int)

        self.gt_unigrams = defaultdict(float)
        self.gt_bigrams = defaultdict(float)
        self.gt_trigrams = defaultdict(float)
        
        self.START_SYMBOL = "<s>"
        self.FIRST_END_SYMBOL = "</s_1>"
        self.SECOND_END_SYMBOL = "</s_2>"
        
    def populate(self):
        with open(self.filename) as f:
            all_words = []
            for line in f:
                line_with_symbols = "%s %s %s %s" % (self.START_SYMBOL, line, self.FIRST_END_SYMBOL, self.SECOND_END_SYMBOL)

                tokens = line_with_symbols.lower().split()
                all_words.extend(tokens)
                # 10 words -> 13 words; Indices 0-9 -> Indices 1-10
                # 10 words -> 13 words; Indices 0-9 -> Indices 1-10 -> New indices 0-12

            # Build unigrams
            for word in all_words:
                if word != self.SECOND_END_SYMBOL:
                    self.unigrams[(word,)] += 1

            for i in range(len(all_words) - 1):
                bigram = (all_words[i], all_words[i+1])
                if self.SECOND_END_SYMBOL not in bigram:
                    self.bigrams[bigram] += 1
                    
            for i in range(len(all_words)-2):
                trigram = (all_words[i], all_words[i+1], all_words[i+2])
                self.trigrams[trigram] +=1


        self.unique_unigrams = len(self.unigrams.keys())
        self.unique_bigrams = len(self.bigrams.keys())
        self.unique_trigrams = len(self.trigrams.keys())
        # GT Linear Regression
        # Get all data points for frequency of frequency i.e 10 words occured once, 3 words occured twice etc.
        # use (1, 10), (2, 3) to create linear model to approximate (c, N)
        
        
        counts_to_fof_unigram = Counter(self.unigrams.values())
        counts_to_fof_bigram = Counter(self.bigrams.values())
        counts_to_fof_trigram = Counter(self.trigrams.values())

        unigram_lr = log_linear_regression(counts_to_fof_unigram)
        bigram_lr = log_linear_regression(counts_to_fof_bigram)
        trigram_lr = log_linear_regression(counts_to_fof_trigram)

        def modified_count(c, lr, defer):
            k = self.GT_UPPER_BOUND
            if c == 0 or c == 1:
                return lr(1)/defer
            elif 1 < c <= k:
                var =  (((c+1) * (lr(c+1)/lr(c))) - (c * (((k+1) * lr(k+1))/lr(1))))/(1 - (((k+1) * lr(k+1))/lr(1)))
                return var
            else:
                return c

        bigram_profile = defaultdict(set)
        
        """
        Bigram profile keys should be unigrams such that 
        profile[the] -> [(the dog), (the cat), (the monkey)]
        so that you can access the number of words w such that c(w|the) > 1
        """
        for key in self.bigrams:
            w1, w2 = key
            bigram_profile[w1].add(key)
            
        self.gt_unigrams = CustomDict(lambda key: modified_count(self.unigrams[key], unigram_lr, self.unique_unigrams))
        self.gt_bigrams = CustomDict(lambda key: modified_count(self.bigrams[key], bigram_lr, self.unique_bigrams))
        self.gt_trigrams = CustomDict(lambda key: modified_count(self.trigrams[key], trigram_lr, self.unique_trigrams))

    
def log_linear_regression(dict_pairs):
    n = len(dict_pairs.keys())
    sum_xy = sum(log(k) * log(v) for k,v in dict_pairs.items())
    sum_x = sum(log(k) for k in dict_pairs)
    sum_y = sum(log(v) for v in dict_pairs.values())
    sum_x2 = sum(log(k) * log(k) for k in dict_pairs)
    m = ((n * sum_xy) - (sum_x * sum_y))/((n * sum_x2) - ((sum_x)**2))
    b = (sum_y - m * sum_x)/n
    return lambda x: exp(b + (m * log(x)))

def gt_prob(inp, fn):
    return exp(fn(log(inp)))


"""
1/10 
2/10
3/10
4/10

(c + 1) / (t + v + 1)
-- 1/15
2/15
3/15
4/15
6/15

the dog, the cat, the cat, the cat, the dog, the monkey, the fish the fish the fish the fish
P(monkey|the) = C(the, monkey)/c(the) = 1/10
P(fish|the) = C(the, fish)/C(the) = 4/10

P(unseen|the) = C(the, unseen) + 1/(C(the) + V + 1) = 1/15
V is the # of unique words w such that (the, w) exists
"""

class CustomDict(dict):
    def __init__(self, factory):
        self.factory = factory
        
    def __missing__(self, key):
        self[key] = self.factory(key)
        return self[key]
    
def gt_prob(x, r):
    return 1
