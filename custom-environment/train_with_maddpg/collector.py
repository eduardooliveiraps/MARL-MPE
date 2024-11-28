from torchrl.collectors import SyncDataCollector
from tensordict.nn import TensorDictSequential

def create_collector(env, exploration_policies, config):
    # Put exploration policies from each group in a sequence
    agents_exploration_policy = TensorDictSequential(*exploration_policies.values())

    collector = SyncDataCollector(
        env,
        agents_exploration_policy,
        device=config.device,
        frames_per_batch=config.frames_per_batch,
        total_frames=config.total_frames,
    )

    return collector