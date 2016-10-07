"""
Probability based metrics
These metrics expect a predicted probability (pred_probs_vec) for each case
And an answer (truth_vec)
"""
import numpy as np

def log_lik(truth_vec,pred_probs_vec):
    """
    TODO:
    Should start by doing some checking
    Assert both are of the same length
    Assert pred_probs_vec and truth_vec are between 0 and 1
    Warn (but not error) if truth_vec has values strictly between 0 and 1
    """
    total_log_lik = np.sum(truth_vec*np.log(pred_probs_vec)+(1-truth_vec)*np.log(1-pred_probs_vec))
    return total_log_lik

def avg_log_lik(truth_vec,pred_probs_vec):
    average_log_lik = log_lik(truth_vec,pred_probs_vec)/len(pred_probs_vec)
    return average_log_lik

def exp_avg_log_lik(truth_vec,pred_probs_vec):
    return np.exp(avg_log_lik(truth_vec,pred_probs_vec))

def binary_entropy_nats(prob):
    return -prob*np.log(prob) - (1-prob)*np.log(1-prob)

def binary_entropy_nats_prime(prob):
    return np.log((1-prob)/prob)

def inverse_binary_entropy_nats(entropy_val, num_iter=3):
    guess = (np.arcsin((entropy_val/np.log(2))**(1/.645)))/np.pi
    for i in range(num_iter):
        guess = guess + np.nan_to_num((entropy_val-binary_entropy_nats(guess))/binary_entropy_nats_prime(guess))
    return guess


def inverse_entropy_norm_lik(truth_vec,pred_probs_vec):
    return 1- inverse_binary_entropy_nats(-avg_log_lik(truth_vec,pred_probs_vec))


def exact_ROC_AUC(truth_vec,score_vec):
    """
    Ranking based metrics
    TODO:
    Should check that truth_vec and score_vec are same length
    Should check that truth_vec is either 1 or 0
    """
    scorevec1 = np.sort(score_vec[truth_vec==1])
    scorevec0 = np.sort(score_vec[truth_vec==0])
    len1 = len(scorevec1)
    len0 = len(scorevec0)
    denom = len1*len0
    i0 = 0
    i1 = 0
    gtnumer = 0
    while i1<len1:
        while ((i0<len0) and (scorevec1[i1]>scorevec0[i0])):
            i0+=1
        gtnumer += (i0)
        i1+=1
    i0 = 0
    i1 = 0
    geqnumer = 0
    while i1<len1:
        while ((i0<len0) and (scorevec1[i1]>=scorevec0[i0])):
            i0+=1
        geqnumer += (i0)
        i1+=1
    return(.5*(gtnumer+geqnumer)/denom)
