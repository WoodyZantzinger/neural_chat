__author__ = 'woodyzantzinger'

import json
import os
import pdb
import nltk

DIR = os.getcwd() + '/slack/'

def slack_load(channels, USER = "U03S8EK4R"):

    #Place to store all the messages found
    response = []
    prompt = []

    for channel in channels:
        for filename in os.listdir(DIR + channel):
            if filename[0] == ".": continue
            with open(DIR + channel + "/" + filename) as data_file:
                data = json.load(data_file)

                for x, message in enumerate(data):
                    #Some messages with Bots do not have a "user" hence the "try"
                    if "user" in message and message["type"] == "message" and message["user"] == "U03S8EK4R":
                        #This is an Andrew Morgan message, was the message a response to another message?

                        prompt_message = data[(x - 1) % len(data)]
                        #pdb.set_trace()

                        if prompt_message != message \
                                and prompt_message["type"] == "message" \
                                and (float(message["ts"]) - float(prompt_message["ts"])) < 500\
                                and message["text"].lower()[0] != "<"\
                                and prompt_message["text"].lower()[0] != "<":

                            #is this message crazy long?
                            p = nltk.sent_tokenize(prompt_message["text"].lower())
                            r = nltk.sent_tokenize(message["text"].lower())
                            if len(p[0]) < 100 and len(r[0]) < 100:
                                #This is a true prompt and response, store both
                                prompt.append(p)
                                response.append(r)

    print "found %d" % len(response)
    
    return [prompt, response]
