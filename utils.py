import numpy as np
import pdb

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def save_model_parameters(outfile, model):
    U, V, W = model.U, model.V, model.W
    np.savez(outfile, U=U, V=V, W=W)
    print "Saved model parameters to %s." % outfile

def load_model_parameters(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U = U
    model.V = V
    model.W = W
    print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1])
