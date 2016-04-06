import nltk
import datetime
import sys
import os
import slack_load
import itertools
import RNNNumpy
import time
import pdb
import theano
import theano.tensor as T
from utils import *
from timeit import default_timer as timer
import datetime

vocabulary_size = 4500
sentence_start_token = "MESSAGE_START"
unknown_token = "UNKNOWN_TOKEN"
sentence_end_token = "MESSAGE_END"
sentence_blank_token = "BLANK"
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))

# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            save_model_parameters("data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


channels = ["pm",
            "cville",
            "hbon",
            "hboespana"]

print "Reading Slack history"
slack_data = slack_load.slack_load(channels)

# Append SENTENCE_START and SENTENCE_END
responses = ["%s %s %s" % (sentence_start_token, x[0], sentence_end_token) for x in slack_data[1]]
prompts = ["%s %s %s" % (sentence_start_token, x[0], sentence_end_token) for x in slack_data[0]]
print "Parsed %d sentences." % (len(responses))

# Tokenize the sentences into words
tokenized_responses = [nltk.word_tokenize(sent) for sent in responses]
tokenized_prompts = [nltk.word_tokenize(sent) for sent in prompts]

# Count the word frequencies
word_freq_prompts = nltk.FreqDist(itertools.chain(*tokenized_prompts))
word_freq_responses = nltk.FreqDist(itertools.chain(*tokenized_responses))
print "Found %d unique words tokens in prompts." % len(word_freq_prompts.items())
print "Found %d unique words tokens in responses." % len(word_freq_responses.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab_prompts = word_freq_prompts.most_common(vocabulary_size-1)
index_to_word_prompts = [x[0] for x in vocab_prompts]
index_to_word_prompts.append(unknown_token)
word_to_index_prompts = dict([(w,i) for i,w in enumerate(index_to_word_prompts)])

vocab_responses = word_freq_responses.most_common(vocabulary_size-1)
index_to_word_responses = [x[0] for x in vocab_responses]
index_to_word_responses.append(unknown_token)
word_to_index_responses = dict([(w,i) for i,w in enumerate(index_to_word_responses)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab_prompts[-1][0], vocab_prompts[-1][1])

# Replace all words not in our vocabulary with the unknown token (Responses)
for i, sent in enumerate(tokenized_responses):
    tokenized_responses[i] = [w if w in word_to_index_responses else unknown_token for w in sent]

# Replace all words not in our vocabulary with the unknown token (Prompts)
for i, sent in enumerate(tokenized_prompts):
    tokenized_prompts[i] = [w if w in word_to_index_prompts else unknown_token for w in sent]

print "\nExample sentence: '%s'" % responses[5]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_responses[5]
 
# Create the training data
X_train = np.asarray([[word_to_index_prompts[w] for w in sent] for sent in tokenized_prompts])
y_train = np.asarray([[word_to_index_responses[w] for w in sent] for sent in tokenized_responses])

for x, prompt in enumerate(X_train):
    dif = len(prompt) - len(y_train[x])
    if dif > 0:
        #prompts is longer, pad responses
        y_train[x] = np.append(y_train[x], [1] * dif)

    elif dif < 0:
        #response is longer, pad prompts
        X_train[x] = np.append(X_train[x], [1] * (dif*-1))

np.random.seed(10)
# Train on a small subset of the data to see what happens
model = RNNNumpy.RNNNumpy(vocabulary_size)
load_model_parameters('data/best.npz', model)

while (1):
    input = raw_input("Ask Andrew Morgan: ")
    input = "%s %s %s" % (sentence_start_token, input, sentence_end_token)
    t_input = nltk.word_tokenize(str(input))
    t_input = [w if w in word_to_index_responses else unknown_token for w in t_input]
    final_input = np.asarray([word_to_index_prompts[w] for w in t_input])
    sentence_probability = model.forward_propagation(final_input)
    pdb.set_trace()

#losses = train_with_sgd(model, X_train, y_train, nepoch=10, evaluate_loss_after=1)

