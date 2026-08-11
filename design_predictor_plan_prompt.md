I want to finish my implementation of src/codesign/hyperdesigners/mo_design_predictor_hypernetwork.py. This script is similar to src/codesign/hyperdesigners/mo_design_hypernetwork.py, except that it uses a design predictor network to sample optimal designs from tradeoffs, and then feeds these designs into the mo design hyperentwork. I would like you to pick up where i left off in my implementation and finish it. First here is a clear description of the network architectures and losses.

1. MO design hypernetwork takes H(d, w) = \pi_{d,w} and V_{d,w}. It is trained by the multi objective PPO loss in src/codesign/hyperdesigners/losses.py. The algorithm works by sampling a grid of tradeoffs and designs, rolling these out, and backproping on the MO PPO loss as mentioned above.

2. Design predictor network. This is a new network that should be f(d | w), where w is a tradeoff and d is a distribution of optimal designs for that tradeoff. In other words, f should produce a mean and logvar of design for any tradeoff w. I want to train f using the GRPO loss. We construct GRPO as follows: a group is a collection of rollouts given a single tradeoff. Each rollout has a different design, stochastically sampled from f(d | w). The values that GRPO computes advantages from are computed from the discounted rollouts within that group. Make sure to handle terminations appropriately. 

Please complete the following tasks

1. Assess whether my current implementation is 'on the right track' given what i have just told you. Let me know if you spot any inconsistencies, issues with normalization, or if I am mis representing the design predictor distribution anywhere

2. Sampling designs using the design predictor: this requires un normalizing outputs of the design prediciton and passing them into generate model.

3. Unpack the data of the design predictor in preparation for updating the design prediction network (i.e. construct the groups, create the values from rollouts, and pass this into GRPO loss and do backprop)

4. SGD on design predictor: please implement the gradient descent procedure using the gradient_update_fn for the design predictor. It should have a similar form as the one for the mo design hypernetwork.

5. Saving the design predictor parameters and writing inference functions for them. You should add this to the rollout_single script in scripts/ and connect it to a CLI argument so that I can (a) use the design predictor if i want and (b) input a specific tradeoff (that might be different than the tradeoff specified for the environment). This would be to visually test alignment and specificity of the design predictor.


Throughout this process, ask me all questions you have. Even if you think you can assume the answer, always ask me first if you are slightly unsure. Also, reuse code that already exists. Do not write 200 lines when something could be written in 50. Refer to CLAUDE.md for more style considerations.