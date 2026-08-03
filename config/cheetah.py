from codesign import config

cheetah_ppo = config.Train(
    name       = 'test',
    save_dir   = 'results/ppo-cheetah',
    algorithm  = 'ppo',
    env        = 'CodesignCheetah',
    backend    = 'jnp',
    env_config = config.Cheetah(
        ctrl_dt      = 0.01,
        sim_dt       = 0.01,
        action_scale = 1.0,
        reward       = config.Reward(
            weights    = config.CheetahWeights(
                run       = 1.0,
                height    = 10.0,
                energy    = 0.00003,
                alive     = 1.0,
                done      = 0
            ),
            objectives = config.Objectives(
                objectives              = [['run'], ['energy'], ['height']],
                shared_objectives       = ['alive'],
                labels                  = ['Running Speed', 'Energy', 'Height'],
                default_scalarization   = [1.0, 1.0, 1.0]
            ),
        ),
        codesign = config.Codesign(
            default_design  = [1.0],
            designs = [
                config.Design(
                    name      = 'front thigh',
                    setter    = lambda cls, spec, scale_factor: cls.change_link_length(spec, 'fthigh', 'fshin', scale_factor),
                    low       = 0.5,
                    high      = 2.0
                ),
            ]
        )
    ),
    learning = config.BasePPO(
        ppo_params = config.PPO(
            num_timesteps          = 50000000,
            episode_length         = 500,
            num_envs               = 1024,
            unroll_length          = 20,
            batch_size             = 64,
            num_minibatches        = 16,
            num_updates_per_batch  = 4,
            learning_rate          = 0.0003,
            entropy_cost           = 0.01,
            discounting            = 0.98,
            reward_scaling         = 1.0,
            clipping_epsilon       = 0.2,
            gae_lambda             = 0.95,
            max_grad_norm          = 1.0,
            normalize_advantage    = True,
            normalize_observations = True,
            num_evals              = 15,
            num_eval_envs          = 256,
            deterministic_eval     = True,
            seed                   = 0,
        ),
        network_params = config.ActorCriticNetworks(
            policy_hidden_layer_sizes = [64, 64],
            value_hidden_layer_sizes  = [64, 64]
        ),
    ),
)
