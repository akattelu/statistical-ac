import sys
from utils import * 
import probabilities
from itertools import chain
from math import log, exp, log2
from collections import defaultdict

def perplexity(model, test_file):
    # Return perplexity of model on the test file
    
    # Infer n-gram number
    first_item = list(model.keys())[0]
    n_grams = len(first_item)
    
    all_words = []
    with open(test_file) as f:
        for line in f:
            if n_grams == 1 or n_grams == 2:
                sent = "%s %s %s" % ("<s>", line, "</s_1>")
            elif n_grams == 3:
                sent = "%s %s %s %s" % ("<s>", line, "</s_1>", "</s_2>")
            else:
                pass
            
            words = sent.lower().split()

            # all_words.append(words)
            all_words.extend(words)


    total = 0
    for i in range(len(all_words) - (n_grams - 1)):
        tup = []
        for x in range(n_grams):
            tup.append(all_words[x+i])
        key = tuple(tup)
        probs = model[key]
        # print(key, probs)
        total += log2(probs)

        
    total_length = len(all_words) - (n_grams - 1)        

    # print(total, total_length)
    total = (-1/total_length) * total

    total = 2**(total)
            
    # total_product = 0
    # for sentence in all_words:
    #     for i in range(len(sentence)-(n_grams - 1)):
    #         # Build key
    #         tup = []
    #         for x in range(n_grams):
    #             tup.append(sentence[x+i])
    #         key = tuple(tup)
    #         probs = model[key]
    #         if probs > .9:
    #             print(key, probs)
    #         total_product += log(probs)



    # total_product = (-1.0/total_length) * (total_product)
    # total_product = exp(total_product)

    
    # return "Perplexity: %f " % (total_product)
    return total
