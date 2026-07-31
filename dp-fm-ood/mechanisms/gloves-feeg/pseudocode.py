"""
pseudocode for feeg blending mechanism

gist is that it's similar to "to the noise and back" (yoneda) in that it starts with some
level of noise (in Yoneda - partially noised action) and work your way backwards (=denoising, =back to user-similar action input)

what is unique about this though is it frames it as energy "guidance" where
the velocity field predicts (?) how close or far its user-similar action input will be after finishing denoising
from some t to t = 1 (1 being timepoint where it is fully denoised from where it started)
and aims to stay close to a_u because gradient of E x1_hat gives distance between endpoint (x1) and a_u
and so you want to minimize that

also another point is that you can't directly get x1 so you treat that the remaining duration (1-t) velocity will be that of 
timestep t and you just approximate

algo**
input: noisy agent action a_u
required: autonomy level t (the bigger, closer to user input; it's the opposite from yoneda, because
this is equiv to having a smaller gamma, meaning that it's not denoised a ton and stays
more closer to the raw user action input; 0 ~ 1), guidance strength gamma_align (0 ~ 1)

(i don't know yet how to deal with - `With OOD detection, an action is refined by FEEG only when
needed. Otherwise, the action is considered in-distribution and executed directly.
this sounds like it requires some <policy> that decides on when to trigger this)`

output: refined action (or action chunk?!)
"""

import numpy as np
import torch

# TODO: write type-checked code

def main(action_chunk, t, gamma_align):
    a_u = action_chunk

    mu = 0; sigma = 1
    noise = np.random.normal(mu, sigma)
    x_t = t * a_u + (1-t) * noise

    # TODO - define `velocity_net` and `context` (see `density_nonconformity_score_calc.py`)
    # velocity_net: dit_flow: DiTFlowModel 's velocity_net
    # context: something like this -- context = dit_flow._prepare_context_tokens(batch)
    velocity_net = ''
    context = ''

    # normalize t into the [B]-shaped tensor forward()/_TimeNetwork expects - see
    # _TimeNetwork.forward's `assert len(t.shape) == 1` and _DiTNoiseNet.sample's `t_all`,
    # built the same way: one value per batch element, already in [0,1] (the *1000 scaling
    # happens inside _TimeNetwork itself, so t must NOT be pre-scaled here)
    t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=x_t.dtype)

    # need grad w.r.t. x_t to get J_h^T (x1_hat - a_u) below via autograd, instead of forming
    # (∂v_θ/∂x_t) as an explicit matrix by hand
    x_t = x_t.requires_grad_(True)
    v = velocity_net(x_t, t_tensor, context)  # v_theta(x_t, t | c), reused for both terms below
    x1_hat = x_t + (1 - t) * v

    # reg_term = ∇_{x_t} E(x1_hat) = (x1_hat - a_u) + (1-t)*(∂v_θ/∂x_t)^T (x1_hat - a_u), all
    # computed in one vector-Jacobian product by autograd.grad - no explicit Jacobian matrix
    energy = 0.5 * (x1_hat - a_u).pow(2).sum()
    reg_term = torch.autograd.grad(energy, x_t)[0]

    refined_chunk = v - gamma_align * reg_term

    return refined_chunk
    

if __name__ == "__main__":

    # NOTE: tweak parameters
    t = 0.5 # note that it's not raw timestep, but normalized between 0 ~ 1
    gamma_align = 1 # per the paper: "Across all experiments, λalign = 1 yields stable and effective performance"

    action_chunk = "" # TODO: fill in with real action_chunk, probably a np array? or a torch tensor

    refined_chunk = main(action_chunk, t, gamma_align)