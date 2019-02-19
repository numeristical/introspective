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


def roc_auc_direct(truth_vec,score_vec):
    """Directly computes the ROC AUC by counting the proportion of 1-0 pairs which are correctly
    ordered.
    """

    if len(score_vec.shape)>1:
        score_vec = score_vec[:,1]

    scorevec1 = np.sort(score_vec[truth_vec==1])
    scorevec0 = np.sort(score_vec[truth_vec==0])
    len1 = len(scorevec1)
    len0 = len(scorevec0)
    denom = len1*len0
    dvi_1 = np.where(np.diff(scorevec1))[0]
    dvi_0 = np.where(np.diff(scorevec0))[0]
    dvi_1 = np.concatenate((dvi_1,[len(scorevec1)-1]))
    dvi_0 = np.concatenate((dvi_0,[len(scorevec0)-1]))

    weight_vec_1 = np.concatenate(([dvi_1[0]+1], np.diff(dvi_1)))
    weight_vec_0 = np.concatenate(([dvi_0[0]+1], np.diff(dvi_0)))
    len_dvi_1 = len(dvi_1)
    len_dvi_0 = len(dvi_0)
    i0,i1 = 0,0
    amassed_weight_0 = 0
    gtnumer, geqnumer = 0,0
    while i1<len_dvi_1:
        amassed_weight_1 = weight_vec_1[i1]
        while ((i0<len_dvi_0) and (scorevec1[dvi_1[i1]]>scorevec0[dvi_0[i0]])):
            #i0+=1
            amassed_weight_0+=weight_vec_0[i0]
            i0+=1                      
        gtnumer += (amassed_weight_0 * amassed_weight_1)
        
        i0a = i0
        amassed_weight_0a = amassed_weight_0
        while ((i0a<len_dvi_0) and (scorevec1[dvi_1[i1]]>=scorevec0[dvi_0[i0a]])):
            #i0+=1
            amassed_weight_0a+=weight_vec_0[i0a]
            i0a+=1            
        geqnumer += (amassed_weight_0a * amassed_weight_1)
        i1+=1
    i0 = 0
    i1 = 0
    return(.5*(gtnumer+geqnumer)/denom)
