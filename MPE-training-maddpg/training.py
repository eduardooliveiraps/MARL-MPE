import torch
from tensordict import TensorDictBase
from tqdm import tqdm
import copy
import calendar
import time
import os
import csv

def process_batch(batch: TensorDictBase, env) -> TensorDictBase:
    for group in env.group_map.keys():
        keys = list(batch.keys(True, True))
        group_shape = batch.get_item_shape(group)
        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )
    return batch
def create_name(config):
    current_GMT = time.gmtime()
    ts = calendar.timegm(current_GMT)
    aux = 'vmas' if config.use_vmas==True else ''
    return f'{config.env_name}_{ts}_{config.algorithm}_{aux}_{config.n_iters}.pth'

def save(policy,config,name):
    directory = f'./models/{config.env_name}'
    os.makedirs(directory, exist_ok=True)
    torch.save(policy,f'{directory}/{name}')

def train(env, collector, replay_buffers, losses, optimizers, target_updaters, exploration_policies, config):
    os.system('cls' if os.name == 'nt' else 'clear')
    print('Training Started')
    name = create_name(config)
    pbar = tqdm(
        total=config.n_iters,
        desc=", ".join(
            [f"episode_reward_mean_{group} = 0" for group in env.group_map.keys()]
        ),
    )
    episode_reward_mean_map = {group: [] for group in env.group_map.keys()}
    train_group_map = copy.deepcopy(env.group_map)
    count=0
    checkpoint_interval = config.n_iters // 10
    results = []

    for iteration, batch in enumerate(collector):
        current_frames = batch.numel()
        batch = process_batch(batch, env)
        for group in train_group_map.keys():
            group_batch = batch.exclude(
                *[
                    key
                    for _group in env.group_map.keys()
                    if _group != group
                    for key in [_group, ("next", _group)]
                ]
            )
            group_batch = group_batch.reshape(-1)
            replay_buffers[group].extend(group_batch)

            for _ in range(config.n_optimiser_steps):
                subdata = replay_buffers[group].sample()
                loss_vals = losses[group](subdata)

                for loss_name in ["loss_actor", "loss_value"]:
                    loss = loss_vals[loss_name]
                    optimizer = optimizers[group][loss_name]

                    loss.backward()

                    params = optimizer.param_groups[0]["params"]
                    torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()

                target_updaters[group].step()

            exploration_policies[group][-1].step(current_frames)
        count+=1
        if hasattr(config, 'iteration_when_stop_training_evaders') and iteration == config.iteration_when_stop_training_evaders:
            del train_group_map["agent"]

        iteration_results = [iteration]
        for group in env.group_map.keys():
            episode_reward_mean = (
                batch.get(("next", group, "episode_reward"))[
                    batch.get(("next", group, "done"))
                ]
                .mean()
                .item()
            )
            episode_reward_mean_map[group].append(episode_reward_mean)
        
        
        pbar.set_description(
            ", ".join(
                [
                      
                      f"episode_reward_mean_{group} = {episode_reward_mean_map[group][-1]}"
                        for group in env.group_map.keys()
   
                ]
            )+"\n...Saving...'" if count % checkpoint_interval == 0 or count == config.n_iters else '',
            refresh=False,
        )
        pbar.update()
        
        if(count % checkpoint_interval == 0 or count == config.n_iters):
            save(collector.policy,config,name)
    save(collector.policy,config,name)
    print("Training finished\nPress enter to continue...")
    input()

    '''
    with open('training_results_iddpg.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Adversary', 'Agent'])
        for result in results:
            writer.writerow(result)
    '''