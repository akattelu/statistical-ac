# Statistical Auto-Complete (SAC)

SAC is primarily a queryable server-side application. It uses bigram
language models to statistically predict what is likely to be a next
word, given the previous word. 

## Getting Started

### Requirements
1. Python 3 (get it
   at [Python Downloads](https://www.python.org/downloads/))
2. Flask library (```pip install flask```)
3. (Optional) The ivy emacs package (```M-x package-install ivy RET```)

### Installation
1. Clone the repository
2. Make sure you have the requirements installed
3. Download some sample text that the model can learn from. Try to get
   3 separate files (one for training, one for testing, and one for
   parameter optimization. You may also use the files provided in the
   penn folder)
4. (Optional) Copy the contents of the lm-ac.el file into your emacs.d
   init file, and bind the lm-ac function to any key binding you want

## Usage
* Running the application with the -h flag displays the help usage text
* Run the server with the following command:
```
python3 model.py --n 2 --smoothing interpolated --train-file penn/train.txt --dev-file penn/valid.txt --test-file penn/test.txt  --action server
```

	Currently the only supported option for n is 2. I am working on
	including 3-5 grams.

* The terminal will hang, and the server will start running.
* Send a get request to localhost:5000/req with the word parameter as
  the "prior" word. For example:
  
  ```
  curl "localhost:5000/req?word=bank"
  ```
  
  The server will respond with json, where the data attribute contains
  a list of words, in order of most likely to follow the prior word.
  
  ```
  {
  "data": [
    "of", 
    "'s", 
    "and", 
    "holding", 
    "in", 
    "to", 
    "the", 
    "said", 
    "debt", 
    "has"
  ]
  }
  ```

* To use the emacs feature, have the server running and then run the
  lm-ac function, with the cursor point following the "prior word"

## Other Functionality 

The application also supports a couple of other features, namely
random sentence generation, and perplexity evaluation.

### Random Sentence Generation

You can run:

```
python3 model.py --n 2 --smoothing interpolated --train-file penn/train.txt --dev-file penn/valid.txt --test-file penn/test.txt  --action generate
```

The model will attempt to generate and print a string of words based on the
probabilities it estimates. 

```

for
that
people
need
as
power
analysts
said
kidder
brokers
a
year
buoyed
by
<unk>
a
gene
taylor
inc.
questioned
the
butler
the
butler
wis.
manufacturer
went
bargain-hunting

```

### Perplexity Evaluation

Running the application with the perplexity action will evaluate the
model on the specified test file, and print out the perplexity.

```
python3 model.py --n 2 --smoothing interpolated --train-file penn/train.txt --dev-file penn/valid.txt --test-file penn/test.txt  --action perplexity
```
p
```
Perplexity 196.01943269418516
```
