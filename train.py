import nltk
import slack_load
import itertools
from utils import *
from random import randint

vocabulary_size = 30000
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

print "Sample call -> responses:"
for _ in range(20):
    ran = randint(0,len(responses))
    print "\t %s \n\t\t %s" % (prompts[ran], responses[ran])

# Count the word frequencies
word_freq_prompts = nltk.FreqDist(itertools.chain(*tokenized_prompts))
word_freq_responses = nltk.FreqDist(itertools.chain(*tokenized_responses))
print "Found %d unique words tokens in prompts." % len(word_freq_prompts.items())
print "Found %d unique words tokens in responses." % len(word_freq_responses.items())

resp_num = len(word_freq_responses.items())

print "Rare Words include (from %d to %d):" % (resp_num - 500, resp_num)
for _ in range(20):
    ran = randint(resp_num - 500,resp_num)
    print "\t %s : %d (pos: %d)" % (word_freq_responses.items()[ran][0], word_freq_responses.items()[ran][1], ran)

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

write_to_file("train/n_data/prompt_vocab.txt", index_to_word_prompts)
write_to_file("train/n_data/response_vocab.txt", index_to_word_responses)

split = int(len(X_train)*.9)

print("Size of train data: %d", len(X_train))
print("Size of development [test] data: %d", len(X_train) - split)

write_to_file("train/n_data/input_train.txt", X_train[:split])
write_to_file("train/n_data/output_train.txt", y_train[:split])

write_to_file("train/n_data/input_dev.txt", X_train[split:])
write_to_file("train/n_data/output_dev.txt", y_train[split:])