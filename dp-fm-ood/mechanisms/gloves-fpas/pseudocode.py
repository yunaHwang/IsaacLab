"""
pseudocode for fpas blending mechanism

gist is that you have n random candidates per each iteration (total: k)'s edited action chunk

progressing from action chunk 1 to 2, let's say works by adding the existing action chunk 1 with some
weighted average value, which consists of a linear combo between the weights and the noises (noises that yield small lossses take up larger-value weights)
and so instead of doing a top-1 or throwing out all the different noises and their respective candidates, you instead weight them to smooth it out
and keep on repeating it to K steps until you have a final action chunk

it's called flow-prior action sampling because the premise is using z, inspecting whether it is near the origin
(=high density) or in the tail area (=low density), and using that "flow-prior" as a proxy to 
gauge whether an action is OOD or not.

algo **
input: non-conformity score s(a)
required: calibrated OOD threshold tau, source consistency gamma_align (0 ~ 1), 
        sampling iteration # K, # of candidates n, edit strength eta (0 ~ 1), variance sigma
output: each iteration's refined action chunk and then eventually a_K (the all, accumulated version)
""" 

import os
import sys

import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "ood-signal", "density"))
from density_nonconformity_score_calc import nonconformity_score

def get_loss_val(a_k_i, tau, gamma_align, a_u):

    # TODO - define `velocity_net` and `context`
    # TODO - check exact math with dimensions
    return max((nonconformity_score(velocity_net, a_k_i, context) - tau), 0) + gamma_align * ((a_k_i - a_u).pow(2).sum())

def main(action_chunk, tau, gamma_align, K, n, eta, sigma):
    a_k = action_chunk # first iteration

    a_u = action_chunk # = a_0

    refined_chunks = [a_k] # a_0, a_1, ..., a_K

    for k in range(K):
        # generate n number of candidates
        perturbs_in_this_k = []
        l_values_in_this_k = []
        for i in range(n):
            # draw isotropic local perturbations and sample random action chunks around a_k
            mu = 0
            perturb = np.random.normal(mu, sigma)
            a_k_i = a_k + perturb
            perturbs_in_this_k.append(a_k_i)

            # calculate loss_val (=violation) with this a_k_i
            l_val = get_loss_val(a_k_i, tau, gamma_align, a_u)
            l_values_in_this_k.append(l_val)

        # --- formula (9): weight each candidate by its violation loss (low loss -> high weight) ---
        l_values_in_this_k = np.array(l_values_in_this_k)
        min_l = l_values_in_this_k.min()

        # subtracting min_l before exp() is just the log-sum-exp stabilization trick - the
        # constant cancels top and bottom, so it doesn't change the resulting weights, only
        # keeps exp() from over/underflowing
        unnormalized_weights = np.exp(-(l_values_in_this_k - min_l))
        weights = unnormalized_weights / unnormalized_weights.sum()

        # a^(k+1) <- a^(k) + eta * sum_i w_i * perturb_i
        weighted_perturb_sum = sum(w * perturb for w, perturb in zip(weights, perturbs_in_this_k))
        a_k = a_k + eta * weighted_perturb_sum

        refined_chunks.append(a_k)

    a_K = a_k
    return a_K, refined_chunks


if __name__ == '__main__':
    # NOTE: tweak parameters
    tau = 1.27 # TODO: fix with real number, e.g., 95 percentile equivalent number in the losses dist
    gamma_align = 0.8
    K = 10
    n = 10
    eta = 0.2
    sigma = 0.1

    action_chunk = "" # TODO: fill in with real action_chunk, probably a np array? or a torch tensor

    a_K, refined_chunks = main(action_chunk, tau, gamma_align, K, n, eta, sigma)