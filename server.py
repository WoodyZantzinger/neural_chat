__author__ = 'woodyzantzinger'

import os
import tensorflow as tf
from flask import Flask, request
import json
from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model
import numpy as np

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "outs/", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
app = Flask(__name__)

en_vocab = {}
rev_fr_vocab = []
model = seq2seq_model.Seq2SeqModel
sess = tf.Session()

def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.en_vocab_size, FLAGS.fr_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    model.saver.restore(session, ckpt.model_checkpoint_path)
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  # Load vocabularies.
  en_vocab_path = "n_data/prompt_vocab.txt"
  fr_vocab_path = "n_data/response_vocab.txt"

  global en_vocab
  global rev_fr_vocab
  en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
  _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)
  return model


def decode(sentence):
    # Get token-ids for the input sentence.
    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)

    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(_buckets))
                   if _buckets[b][0] > len(token_ids)])

    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
      {bucket_id: [(token_ids, [])]}, bucket_id)

    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, True)

    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
    # Return sentence corresponding to outputs.
    return (" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))

@app.route("/")
def hello():
    return "Hello world!"

@app.route('/message/', methods=['POST'])
def message():
    print "POST happened"
    #pdb.set_trace()
    return_str = decode(request.form['text'].lower().split("ambot ")[1])
    print "Decode finished"
    resp = {"text": return_str}
    return json.dumps(resp)

if __name__ == "__main__":
    #tf.app.run()
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
