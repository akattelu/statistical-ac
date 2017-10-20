from collections import defaultdict

def file_to_sentences(input_file):
    """
    Splits the input file into a list of strings.
    Strings will be delimited by "." symbol.
    Because the file will be split by periods, the function
    will read the entire file into memory and then parse.
    """
    with open(input_file, 'r') as f:
        return f.read().split(".")

def sentence_to_words(input_str, tags=True):
    """
    Splits a sentence string into separate words.
    Splits by whitespace into list of strings.
    If tags flag is enabled, <s> and </s> as start and end
    tags will be added to the start and end of word list.
    """
    words = input_str.split()
    if tags:
        words.insert(0, "<s>")
        words.append("</s>")
    return words

def make_bigram_count_dict(words_matrix):
    """
    Given a 2D array of words, returns a count dictionary.
    The keys will be tuples of strings indicating the words.
    The values will be the number of times the pair was found.
    """
    counts = defaultdict(int)
    for sentence in words_matrix:
        for i in range(len(sentence) - 1):
            pair = (sentence[i], sentence[i+1])
            counts[pair] += 1

    return counts


def make_unigram_count_dict(words_matrix):
    """
    Given a 2D array of words, returns a count dictionary.
    The keys will be strings referring to each word.
    The values will be the number of times the word was found.
    """
    counts = defaultdict(int)
    for sentence in words_matrix:
        for word in sentence:
            counts[word] += 1

    return counts
