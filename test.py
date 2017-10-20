import unittest
import count_dict, evaluator, probabilities
from collections import Counter
import os
from matplotlib import pyplot
from math import log, exp
import numpy
from evaluator import perplexity
class TestSmoothingMethods(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        import tempfile
        # cls.test_sentence = "the cat jumped over the cat cat big hat dog dog dog dog"
        # cls.test_sentence = "carp carp carp carp carp carp carp carp carp carp perch perch perch whitefish whitefish trout salmon eel"
        cls.test_sentence = "the dog the cat the cat the cat the dog the monkey the fish the fish the fish the fish \n i hate monkeys and rats"
        cls.temp_file = tempfile.NamedTemporaryFile(mode='w+',delete=False)
        cls.temp_file.write(cls.test_sentence)
        cls.temp_file.close()

        # cls.temp_file = open("penn/train.txt")
        # cls.temp_file.close()
        
        cls.count = count_dict.CountDict(cls.temp_file.name)
        cls.count.populate()
        
    @classmethod
    def tearDownClass(cls):
        os.remove(cls.temp_file.name)
        pass
    
    def test_count_dict_unigrams(self):
        with open(self.temp_file.name) as f:
            unigrams = []
            words = []
            for line in f:
                words.append("<s>")
                words.extend(line.lower().split())
                words.append("</s_1>")

            for word in words:
                unigrams.append((word,))
            cd = Counter(unigrams)
            
        for key, value in self.count.unigrams.items():
            self.assertEqual(cd[key], value)

    def test_count_dict_bigrams(self):
        with open(self.temp_file.name) as f:
            bigrams = []

            words = []
            for line in f:
                words.append("<s>")
                words.extend(line.lower().split())
                words.append("</s_1>")
            for i in range(len(words) - 1):
                bigrams.append((words[i], words[i+1]))
            cd = Counter(bigrams)
        for key, value in self.count.bigrams.items():
            self.assertEqual(cd[key], value)

    def test_count_dict_trigrams(self):
        with open(self.temp_file.name) as f:
            trigrams = []

            words = []
            for line in f:
                words.append("<s>")
                words.extend(line.lower().split())
                words.append("</s_1>")
                words.append("</s_2>")

            for i in range(len(words) - 2):
                trigrams.append((words[i], words[i+1], words[i+2]))
            cd = Counter(trigrams)
        for key, value in self.count.trigrams.items():
            self.assertEqual(cd[key], value)

    def test_laplace_bigrams(self):
        probs = probabilities.ProbabilityDict(self.count)
        mle = probs.bigram_MLE()
        laplace = probs.bigram_laplace()
        # for key, value in laplace.items():
            # print("%s given %s is %f" % (key[1], key[0], value))
        self.assertEqual(self.count.unigrams[("the",)], 10)
        self.assertAlmostEqual(laplace[("the", "fish")], 5/15)
        self.assertAlmostEqual(laplace[("the", "monkey")], 2/15)
        self.assertAlmostEqual(laplace[("the", "frog")], 1/15)
        self.assertAlmostEqual(laplace[("the", "chicken")], 1/15)
        self.assertAlmostEqual(laplace[("dog", "the")], 3/4)
        self.assertAlmostEqual(laplace[("dog", "cat")], 1/4)
        self.assertAlmostEqual(mle[("i", "hate")], 1)        
        self.assertAlmostEqual(laplace[("i", "hate")], 2/3)
        self.assertAlmostEqual(laplace[("<s>", "the")], 2/5)        


    def test_laplace_bigram_perplexity(self):
        probs = probabilities.ProbabilityDict(self.count)
        laplace = probs.bigram_laplace()
        p = perplexity(laplace, self.temp_file.name)

        total = 0
        all_words = []
        with open(self.temp_file.name) as f:
            for line in f:
                l = "%s %s %s" % ("<s>", line, "</s_1>")
                tokens = l.lower().split()
                all_words.extend(tokens)

        for i in range(len(all_words) - 1):
            key = (all_words[i], all_words[i+1])
            probs = laplace[key]
            total += log(probs)

        total = total * (-1/((len(all_words) -1)))
        total = exp(total)
        
        self.assertAlmostEqual(total, p)
            
