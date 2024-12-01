from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer

def create_replay_buffers(env, config):
    replay_buffers = {}
    for group, _agents in env.group_map.items():
        replay_buffer = ReplayBuffer(
            storage=LazyMemmapStorage(config.memory_size), # We will store up to memory_size multi-agent transitions
            sampler=RandomSampler(),
            batch_size=config.train_batch_size, # We will sample batches of this size
        )
        if config.device.type != "cpu":
            replay_buffer.append_transform(lambda x: x.to(config.device))
        replay_buffers[group] = replay_buffer

    return replay_buffers