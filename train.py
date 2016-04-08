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
import operator
import heapq

vocabulary_size = 20000
sentence_start_token = "MESSAGE_START"
unknown_token = "UNKNOWN_TOKEN"
sentence_end_token = "MESSAGE_END"
sentence_blank_token = "BLANK"

def write_to_file(file, array):
    with open(file, "w+") as to_write:
        for x in array:
            if type(x) is list:
                for num in x:
                    to_write.write("%s " % num)
                to_write.write("\n")
            else:
                to_write.write("%s\n" % x.encode("UTF-8"))


channels = ["pm",
            "cville",
            "hbon",
            "cbc",
            "teachstone",
            "wyndham",
            "announcements",
            "android",
            "ios"
            ]

#channels = ["test"]
print "Reading Slack history"
slack_data = slack_load.slack_load(channels)

# Append SENTENCE_START and SENTENCE_END
responses = ["%s %s %s" % ("", x[0], "") for x in slack_data[1]]
prompts = ["%s %s %s" % ("", x[0], "") for x in slack_data[0]]
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
vocab_prompts = word_freq_prompts.most_common(vocabulary_size-2)
index_to_word_prompts = [x[0] for x in vocab_prompts]
index_to_word_prompts.append(unknown_token)
index_to_word_prompts.append(sentence_blank_token)
word_to_index_prompts = dict([(w,i) for i,w in enumerate(index_to_word_prompts)])

vocab_responses = word_freq_responses.most_common(vocabulary_size-2)
index_to_word_responses = [x[0] for x in vocab_responses]
index_to_word_responses.append(unknown_token)
index_to_word_responses.append(sentence_blank_token)
word_to_index_responses = dict([(w,i) for i,w in enumerate(index_to_word_responses)])

print "Using vocabulary size %d." % vocabulary_size
#print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab_prompts[-1][0], vocab_prompts[-1][1])

# Replace all words not in our vocabulary with the unknown token (Responses)
for i, sent in enumerate(tokenized_responses):
    tokenized_responses[i] = [w if w in word_to_index_responses else unknown_token for w in sent]

# Replace all words not in our vocabulary with the unknown token (Prompts)
for i, sent in enumerate(tokenized_prompts):
    tokenized_prompts[i] = [w if w in word_to_index_prompts else unknown_token for w in sent]

#print "\nExample sentence: '%s'" % responses[5]
#print "\nExample sentence after Pre-processing: '%s'" % tokenized_responses[5]
 
# Create the training data
X_train = np.asarray([[word_to_index_prompts[w] for w in sent] for sent in tokenized_prompts])
y_train = np.asarray([[word_to_index_responses[w] for w in sent] for sent in tokenized_responses])

max_len = max(len(max(tokenized_prompts, key=len)), len(max(tokenized_responses, key=len)))

write_to_file("n_data/prompt_vocab.txt", index_to_word_prompts)
write_to_file("n_data/response_vocab.txt", index_to_word_responses)

split = len(X_train) - 40000

write_to_file("n_data/input_train.txt", X_train[:split])
write_to_file("n_data/output_train.txt", y_train[:split])

write_to_file("n_data/input_dev.txt", X_train[split:])
write_to_file("n_data/output_dev.txt", y_train[split:])


'''
#Pad everything to max length with Blank tokens
for x, prompt in enumerate(X_train):
    dif = max_len - len(prompt)
    if dif > 0:
        X_train[x] = np.append(X_train[x], [word_to_index_prompts[sentence_blank_token]] * dif)

for x, response in enumerate(y_train):
    dif = max_len - len(response)
    if dif > 0:
        y_train[x] = np.append(y_train[x], [word_to_index_responses[sentence_blank_token]] * dif)

np.random.seed(10)
model = RNNNumpy.RNNNumpy(vocabulary_size)

losses = train_with_sgd(model, X_train, y_train, nepoch=100, evaluate_loss_after=1)

load_model_parameters('data/best.npz', model)


while (1):

    input = raw_input("Ask Andrew Morgan: ")

    #Perform all the formatting on the input
    input = "%s %s %s" % (sentence_start_token, input, sentence_end_token)
    t_input = nltk.word_tokenize(str(input))
    t_input = [w if w in word_to_index_prompts else unknown_token for w in t_input]
    final_input = np.asarray([word_to_index_prompts[w] for w in t_input])

    #Padding
    dif = max_len - len(final_input)
    if dif > 0:
        final_input = np.append(final_input, [word_to_index_responses[sentence_blank_token]] * dif)

    sentence_probability = model.forward_propagation(final_input)[0]

    final_formatted_sentence = ""

    for word in sentence_probability:
        top_results = heapq.nlargest(3, enumerate(word), key=lambda x: x[1])
        final_formatted_sentence += index_to_word_responses[top_results[0][0]] + " "

        for result in top_results:
            print index_to_word_responses[result[0]]

        if top_results[0][0] == word_to_index_responses[sentence_end_token]:
            break

    #pdb.set_trace()

    print "RESULTS:"
    print final_input
    print final_formatted_sentence
'''