__author__ = 'woodyzantzinger'

import os
import time
import sys
from flask import Flask, request

DEBUG = False

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello world!"

@app.route('/message/', methods=['POST'])
def message():
    new_message = request.get_json(force=True)
    print("received message: ")
    print(new_message["text"])
    return 'OK'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "debug":
            DEBUG = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
