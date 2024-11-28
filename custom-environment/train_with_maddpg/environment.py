from torchrl.envs import PettingZooEnv, TransformedEnv, RewardSum, VmasEnv

def create_env(config):
    if not config.use_vmas:
        base_env = PettingZooEnv(
            task="simple_tag_v3",
            parallel=True, # Use the Parallel version
            seed=config.seed,
            # Scenario specific
            continuous_actions=True,
            num_good=config.n_evaders,
            num_adversaries=config.n_chasers,
            num_obstacles=config.n_obstacles,
            max_cycles=config.max_steps,
        )
    else:
        num_vmas_envs = config.frames_per_batch // config.max_steps
        # Number of vectorized environments. frames_per_batch collection will be divided among these environments
        base_env = VmasEnv(
            scenario="simple_tag",
            num_envs=num_vmas_envs,
            continuous_actions=True,
            max_steps=config.max_steps,
            device=config.device,
            seed=config.seed,
            # Scenario specific
            num_good_agents=config.n_evaders,
            num_adversaries=config.n_chasers,
            num_landmarks=config.n_obstacles,
        )

    env = TransformedEnv(
        base_env,
        RewardSum(
            in_keys=base_env.reward_keys,
            reset_keys=["_reset"] * len(base_env.group_map.keys()),
        ),
    )
    return env