__author__ = 'woodyzantzinger'

import json
import os
import pdb
import nltk

DIR = os.getcwd() + '/slack/'

def strip_junk(str):
    final_str = str

    if str.find("@") > -1:
        at_pos = str.find("@")
        space_pos = str.find(" ", at_pos)
        if (at_pos == 0) and (space_pos > 0):
            final_str = str[space_pos+1:]
        elif (at_pos == 0):
            #there was a @ but no space so the @ was the only message
            final_str = ""

    if final_str.find(": ") > -1:
        pos = final_str.find(": ")
        final_str = final_str[pos+2:]
    return final_str



#def slack_load(channels, USER = "U03S8EK4R"):
def slack_load(channels, USER = "U03S8EK4R"):

    #Place to store all the messages found
    response = []
    prompt = []

    rem = {}
    rem["invalid"] = 0
    rem["time"] = 0
    rem["user"] = 0
    rem["text"] = 0
    rem["stripped"] = 0
    rem["callout"] = 0

    channels = os.walk(DIR).next()[1]

    for channel in channels:
        if channel[0] == ".": continue
        for filename in os.listdir(DIR + channel):
            if filename[0] == ".": continue
            with open(DIR + channel + "/" + filename) as data_file:
                data = json.load(data_file)

                carry_prompt = ""

                for x, message in enumerate(data):

                    prev_message = data[(x - 1) % len(data)]

                    #Test Validity of msg and response
                    if not ("user" in message and "text" in message and message["type"] == "message" and len(message["text"]) > 0 and message["text"].lower()[0] != "<")\
                            or not ("user" in prev_message and "text" in prev_message and prev_message["type"] == "message" and len(prev_message["text"]) > 0 and prev_message["text"].lower()[0] != "<"): # and message["user"] == "U03S8EK4R":
                        rem["invalid"] += 1
                        continue

                    #Are these messages close enough together. Current length is 12 seconds?
                    if ((float(message["ts"]) - float(prev_message["ts"])) > 20):
                        rem["time"] += 1
                        continue

                    #If both messages are sent by the same user, we should take the old prompt and save it to add to the next prompt?
                    if (message["user"] == prev_message["user"]):
                        carry_prompt += prev_message["text"] + " "
                        rem["user"] += 1
                        continue

                    #Strip out @ messages
                    if (message["text"].find("@") + message["text"].find(": ") >= 0) or (prev_message["text"].find("@")or prev_message["text"].find(": ") >= 0):
                        message["text"] = strip_junk(message["text"])
                        rem["stripped"] += 1
                        prev_message["text"] = strip_junk(prev_message["text"])
                        if (prev_message["text"] == "" or message["text"] == ""):
                            rem["callout"] += 1
                            continue

                    #Test text of both messages
                    if (len(message["text"].split()) > 25) or (len((carry_prompt + prev_message["text"]).split()) > 25):
                        #Wipe any history we made in the last steps...
                        carry_prompt = ""
                        rem["text"] += 1
                        continue

                    #Add to final lists
                    p = nltk.sent_tokenize((carry_prompt + prev_message["text"]).lower())
                    r = nltk.sent_tokenize(message["text"].lower())
                    prompt.append(p)
                    response.append(r)

                    #Clear backlog
                    carry_prompt = ""

    print "found %d" % len(response)
    print "Removed:"
    print "\t Invalid Message: %d" % rem["invalid"]
    print "\t Messages too far apart: %d" % rem["time"]
    print "\t Messages by same user: %d" % rem["user"]
    print "\t Text too long: %d" % rem["text"]
    print "\t Messages stripped of @: %d" % rem["stripped"]
    print "\t Messages removed for being just a callout: %d" % rem["callout"]
    
    return [prompt, response]
