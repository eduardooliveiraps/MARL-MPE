import torch
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhDelta, AdditiveGaussianModule
from tensordict.nn import TensorDictModule, TensorDictSequential

def create_policies(env, config):
    policy_modules = {}
    for group, agents in env.group_map.items():
        share_parameters_policy = True # Can change this based on the group

        policy_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec[group, "observation"].shape[-1], # n_obs_per_agent
            n_agent_outputs=env.full_action_spec[group, "action"].shape[-1], # n_actions_per_agents
            n_agents=len(agents), # Number of agents in the group
            centralised=False, # the policies are decentralised (i.e., each agent will act from its local observation)
            share_params=share_parameters_policy,
            device=config.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        # Wrap the neural network in a :class:`~tensordict.nn.TensorDictModule`.
        # This is simply a module that will read the ``in_keys`` from a tensordict, feed them to the
        # neural networks, and write the
        # outputs in-place at the ``out_keys``.

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[(group, "observation")],
            out_keys=[(group, "param")],
        ) # We just name the input and output that the network will read and write to the input tensordict
        policy_modules[group] = policy_module

    policies = {}
    for group, _agents in env.group_map.items():
        policy = ProbabilisticActor(
            module=policy_modules[group],
            spec=env.full_action_spec[group, "action"],
            in_keys=[(group, "param")],
            out_keys=[(group, "action")],
            distribution_class=TanhDelta,
            distribution_kwargs={
                "low": env.full_action_spec[group, "action"].space.low,
                "high": env.full_action_spec[group, "action"].space.high,
            },
            return_log_prob=False,
        )
        policies[group] = policy

    return policies, policy_modules

def create_exploration_policies(policies, env, config):
    exploration_policies = {}
    for group, _agents in env.group_map.items():
        exploration_policy = TensorDictSequential(
            policies[group],
            AdditiveGaussianModule(
                spec=policies[group].spec,
                annealing_num_steps=config.total_frames, # Number of frames after which sigma is sigma_end
                action_key=(group, "action"),
                sigma_init=0.9, # Initial value of the sigma
                sigma_end=0.1, # Final value of the sigma
            ),
        )
        exploration_policies[group] = exploration_policy

    return exploration_policies