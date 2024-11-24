import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import custom_environment_v0
import time
import glob
import os


def train_custom_environment(
    env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):
    """
    Trains agents in the custom MPE environment using PPO.
    """
    # Create the environment
    env = env_fn.parallel_env(**env_kwargs)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        learning_rate=1e-3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)

    # Create a folder to save the model if it doesn't exist
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}.zip")

    model.save(model_path)

    print(f"Model has been saved to {model_path}.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    # Wrap the environment using supersuit
    env = ss.pad_observations_v0(env)

    try:
        latest_policy = max(
            glob.glob(f"models/{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    env_fn = custom_environment_v0
    env_kwargs = {}

    train_custom_environment(env_fn, steps=50_000, seed=0, **env_kwargs)

    eval(env_fn, num_games=5, render_mode=None, **env_kwargs)