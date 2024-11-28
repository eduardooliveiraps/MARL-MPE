import torch
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators

def create_losses(policies, critics, env, config):
    losses = {}
    for group, _agents in env.group_map.items():
        loss_module = DDPGLoss(
            actor_network=policies[group], # Use the non-explorative policies
            value_network=critics[group],
            delay_value=True, # Whether to use a target network for the value
            loss_function="l2",
        )
        loss_module.set_keys(
            state_action_value=(group, "state_action_value"),
            reward=(group, "reward"),
            done=(group, "done"),
            terminated=(group, "terminated"),
        )
        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=config.gamma)

        losses[group] = loss_module

    return losses

def create_optimizers(losses, config):
    optimizers = {
        group: {
            "loss_actor": torch.optim.Adam(
                loss.actor_network_params.flatten_keys().values(), lr=config.lr
            ),
            "loss_value": torch.optim.Adam(
                loss.value_network_params.flatten_keys().values(), lr=config.lr
            ),
        }
        for group, loss in losses.items()
    }

    return optimizers

def create_target_updaters(losses, config):
    target_updaters = {
        group: SoftUpdate(loss, tau=config.polyak_tau) for group, loss in losses.items()
    }

    return target_updaters