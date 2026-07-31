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
more closer to the raw user action input), guidance strength gamma_align

(i don't know yet how to deal with - `With OOD detection, an action is refined by FEEG only when
needed. Otherwise, the action is considered in-distribution and executed directly.
this sounds like it requires some <policy> that decides on when to trigger this)`

output: refined action (or action chunk?!)
"""