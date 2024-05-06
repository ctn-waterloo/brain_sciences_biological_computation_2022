import scipy.special
from scipy.special import log_softmax
import numpy as np


## Convert sparsity parameter to neuron bias/intercept
def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))


#def softmax(x):
#    """Compute softmax values for each sets of scores in x."""
#    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

##Softmax Function used for selecting next action
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    filtered_x = np.nan_to_num(x-x.max()) 
    return np.exp(log_softmax(filtered_x))