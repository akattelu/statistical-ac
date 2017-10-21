import pickle
from count_dict import CountDict
from utils import * 
from probabilities import ProbabilityDict
from argparse import ArgumentParser
import evaluator
import random
from flask import Flask, request, jsonify

def main():
    # parser = ArgumentParser(description="Language model generator")
    # parser.add_argument

    parser = ArgumentParser()
    app = Flask(__name__)

    smoothing_choices = ["laplace", "laplace_gt", "gt_mle", "mle", "interpolated", "stupid", "katz"]
    action_choices = ["perplexity", "generate", "server"]
    parser.add_argument("--n", type=int, help="Number of n-grams to use (default: bigrams)", default=2, choices = [1,2,3])
    parser.add_argument("--smoothing", type=str, help="Which type of smoothing to use default laplace", default="laplace", choices=smoothing_choices)
    parser.add_argument("--train-file", type=str, help="File for training the language model", required=True, dest="train")
    parser.add_argument("--test-file", type=str, help="File for testing to generate perplexity", required=True, dest="test")
    parser.add_argument("--dev-file", type=str, help="File for tuning model hyperparameters", required=True, dest="dev")
    parser.add_argument("--action", type=str, help="Action to run", default="perplexity", choices=action_choices)
    args = parser.parse_args()
    
    counts = CountDict(args.train)
    counts.populate()
    
    probs = ProbabilityDict(counts)

    if args.smoothing == "mle":
        if args.n == 1:
            choice = probs.unigram_MLE()
        elif args.n == 2:
            choice = probs.bigram_MLE()
        else:
            pass
    if args.smoothing == "laplace":
        if args.n == 1:
            choice = probs.unigram_laplace()
        elif args.n == 2:
            choice = probs.bigram_laplace()
        else:
            pass
    elif args.smoothing == "laplace_gt":
        if args.n == 1:
            choice = probs.unigram_gt_laplace()
        elif args.n == 2:
            choice = probs.bigram_gt_laplace()
        else:
            pass
    elif args.smoothing == "gt_mle":
        if args.n == 1:
            choice = probs.unigram_gt_MLE()
        elif args.n == 2:
            choice = probs.bigram_gt_MLE()
        else:
            pass
    elif args.smoothing == "interpolated":
        if args.n == 2:
            choices = [x/10 for x in range(1, 10)]
            weights = [(val, evaluator.perplexity(probs.interpolated_bigram(val), args.dev)) for val in choices]
            optimal = min(weights, key=lambda x: x[1])
            print("Optimal weight %f with perplexity %f" % (optimal[0], optimal[1]))
            choice = probs.interpolated_bigram(optimal[0])
        else:
            pass
    elif args.smoothing == "stupid":
        if args.n == 2:
            choice = probs.stupid_backoff_bigram()
        else:
            pass
        
    else:
        pass

    bigram_profile = defaultdict(list)
    for key in counts.bigrams.keys():
        w1, w2 = key
        bigram_profile[w1].append(key)
    
    if args.action == "perplexity":
        print("Perplexity", evaluator.perplexity(choice, args.test))
    elif args.n == 2 and args.action == "generate":
        word = "<s>"
        while word != "</s_1>":
            print(word)
            word = choose_random(word, bigram_profile, choice)
    elif args.action == "server":
        @app.route("/req")
        def req():
            word = request.args.get("word")
            choices = [(key[1],choice[key]) for key in bigram_profile[word]]
            choices = sorted(choices, key=lambda x: x[1], reverse=True)
            filtered = list(filter(lambda x: "<unk>" not in x and "<s>" not in x and "</s_1>" not in x, choices))[:10]
            filtered = [x[0] for x in filtered]
            return jsonify({"data":filtered})
            
        
        app.run(debug=True)

def choose_random(first_word, profile, lm):
    choices = profile[first_word]
    probs = [lm[x] for x in choices]
    prb = random.random()
    for i in range(len(probs)):
        prb -= probs[i]
        if prb <= 0:
            break

    return choices[i][1]
    
if __name__ == '__main__':
    main()

