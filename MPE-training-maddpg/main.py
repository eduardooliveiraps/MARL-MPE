import sys
import copy
import torch
from tqdm import tqdm
from config import Config
from environment import create_env
from policy import create_policies, create_exploration_policies
from critic import create_critics
from collector import create_collector
from replay_buffer import create_replay_buffers
from loss import create_losses, create_optimizers, create_target_updaters
from training import process_batch, train

def main(env_name):
    config = Config(env_name)
    env = create_env(config)
    policies, policy_modules = create_policies(env, config)
    exploration_policies = create_exploration_policies(policies, env, config)
    critics = create_critics(env, config)
    
    reset_td = env.reset()
    for group, _agents in env.group_map.items():
        print(
            f"Running value and policy for group '{group}':",
            critics[group](policies[group](reset_td)),
        )

    collector = create_collector(env, exploration_policies, config)
    replay_buffers = create_replay_buffers(env, config)
    losses = create_losses(policies, critics, env, config)
    optimizers = create_optimizers(losses, config)
    target_updaters = create_target_updaters(losses, config)

    train(env, collector, replay_buffers, losses, optimizers, target_updaters, exploration_policies, config)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <environment_name>")
        sys.exit(1)
    env_name = sys.argv[1]
    main(env_name)