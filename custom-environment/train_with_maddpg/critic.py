import torch
from torchrl.modules import MultiAgentMLP
from tensordict.nn import TensorDictModule, TensorDictSequential

def create_critics(env, config):
    critics = {}
    for group, agents in env.group_map.items():
        share_parameters_critic = True # Can change for each group
        MADDPG = True # IDDPG if False, can change for each group

        # This module applies the lambda function: reading the action and observation entries for the group
        # and concatenating them in a new ``(group, "obs_action")`` entry
        cat_module = TensorDictModule(
            lambda obs, action: torch.cat([obs, action], dim=-1),
            in_keys=[(group, "observation"), (group, "action")],
            out_keys=[(group, "obs_action")],
        )

        critic_module = TensorDictModule(
            module=MultiAgentMLP(
                n_agent_inputs=env.observation_spec[group, "observation"].shape[-1]
                + env.full_action_spec[group, "action"].shape[-1],
                n_agent_outputs=1, # 1 value per agent
                n_agents=len(agents),
                centralised=MADDPG,
                share_params=share_parameters_critic,
                device=config.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            in_keys=[(group, "obs_action")],  # Read ``(group, "obs_action")``
            out_keys=[(group, "state_action_value")], # Write ``(group, "state_action_value")``
        )

        critics[group] = TensorDictSequential(cat_module, critic_module) # Run them in sequence

    return critics