"""
# Custom Environment

|--------------------|------------------------------------------------------------|
| Actions            | Discrete/Continuous                                        |
| Parallel API       | Yes                                                        |
| Manual Control     | No                                                         |
| Agents             | `agents= [agent_0, agent_1, agent_2, adversary_0]` |
| Agents             | 4                                                          |
| Action Shape       | (5)                                                        |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (50))                            |
| Observation Shape  | (14),(16)                                                  |
| Observation Values | (-inf,inf)                                                 |
| State Shape        | (62,)                                                      |
| State Values       | (-inf,inf)                                                 |
|--------------------|------------------------------------------------------------|

The good agents (bluish) are slower and are tasked with intercepting the adversary (reddish). 
The adversary is faster and is rewarded for reaching landmarks (greenish) without being detected by the agents. 
The environment includes obstacles (grayish) that block movement.
By default, there are 3 agents, 1 adversary, 3 landmarks, and 2 obstacles.

So that adversaries don't run to infinity, they are also penalized for exiting the area by the following function:

``` python
def bound(x):
      if x < 0.9:
          return 0
      if x < 1.0:
          return (x - 0.9) * 10
      return min(np.exp(2 * x - 2), 10)
```

Agent and adversary observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]`

Agent and adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python	
custom_environment.env(num_good=3, num_adversaries=1, num_obstacles=2, num_landmarks=3, max_cycles=25, continuous_actions=False)
```

`num_good`:  number of good agents

`num_adversaries`:  number of adversaries

`num_obstacles`:  number of obstacles

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good=3,
        num_adversaries=1,
        num_obstacles=2,
        num_landmarks=3,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            num_landmarks=num_landmarks,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_good, num_adversaries, num_obstacles, num_landmarks)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "custom_environment_v0"

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

class Scenario(BaseScenario):
    def make_world(self, num_good=3, num_adversaries=1, num_obstacles=2, num_landmarks=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_obstacles = num_obstacles
        num_landmarks = num_landmarks
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.05 if agent.adversary else 0.075
            agent.accel = 4.0 if agent.adversary else 3.0
            agent.max_speed = 1.3 if agent.adversary else 1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark_{i}"
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.065
            landmark.boundary = False
        world.obstacles = [Landmark() for i in range(num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = f"obstacle_{i}"
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = 0.2
            obstacle.boundary = True
        world.landmarks += world.obstacles
        return world
    
    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.95, 0.45, 0.45])
                if agent.adversary
                else np.array([0.15, 0.15, 0.65])
            )
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.45, 0.95, 0.45])
        for i, obstacle in enumerate(world.obstacles):
            obstacle.color = np.array([0.25, 0.25, 0.25])
        goal = np_random.choice(world.landmarks)
        goal.color = np.array([0.15, 0.65, 0.15])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.goal_a = goal
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, obstacle in enumerate(world.obstacles):
            obstacle.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            obstacle.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            # Count how many times the adversaries have touched the landmarks
            landmarks_reached = 0
            for landmark in world.landmarks:
                if self.is_collision(agent, landmark):
                    landmarks_reached += 1
            return landmarks_reached
        else:
            # Count how many times the good agent intercepted the adversary
            interceptions = 0
            for adversary in self.adversaries(world):
                if self.is_collision(agent, adversary):
                    interceptions += 1
            return interceptions
        
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
    
    def reward(self, agent, world):
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward
    
    def agent_reward(self, agent, world):
        # Good agents are rewarded for catching adversaries and penalized if adversaries reach landmarks
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        good_agents = self.good_agents(world)
        
        # Optional shaping reward based on distance to adversaries
        if shape:
            for adv in adversaries:
                rew += 0.1 * min(
                    np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
                    for agent in good_agents
                )
        
        # Positive reward for catching adversaries
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew += 20
        
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = sum(
                np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                for a in adversary_agents
            )
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0
            for a in adversary_agents:
                if (
                    np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                    < 2 * a.goal_a.size
                ):
                    adv_rew -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                for a in good_agents
            )
        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            if (
                min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                    for a in good_agents
                )
                < 2 * agent.goal_a.size
            ):
                pos_rew += 5
            pos_rew -= min(
                np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                for a in good_agents
            )
        

        return rew + pos_rew + adv_rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for reaching landmarks and negatively rewarded if caught by good agents
        rew = 0
        shaped_reward = True
        good_agents = self.good_agents(world)
        shaped_reward_value = 0
        adv_rew = 0
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            aux = -np.sqrt(
                np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
            )
        else:  # proximity-based reward (binary)
            if (
                np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
                < 2 * agent.goal_a.size
            ):
                adv_rew += 5

        # Negative reward if caught by good agents
        for good_agent in good_agents:
            if self.is_collision(agent, good_agent):
                rew -= 20  # Reduced negative reward for being caught by a good agent

        # Optionally, shape the reward based on distance to the nearest landmark
        if shaped_reward:
            min_distance = min(
                np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
                for landmark in world.landmarks
            )
            rew -= 0.05 * min_distance  # Reduced negative reward proportional to the distance to the nearest landmark
        
            
        return rew + adv_rew + shaped_reward_value
    """
    def agent_reward(self, agent, world):
        # Good agents are rewarded for catching adversaries and penalized if adversaries reach landmarks
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        good_agents = self.good_agents(world)
        
        # Optional shaping reward based on distance to adversaries
        if shape:
            for adv in adversaries:
                rew += 0.2 * min(
                    np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
                    for agent in good_agents
                )
        
        # Positive reward for catching adversaries
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew += 20
        
        # Negative reward if adversaries reach landmarks
        for adv in adversaries:
            for landmark in world.landmarks:
                if self.is_collision(adv, landmark):
                    rew -= 10

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for reaching landmarks and negatively rewarded if caught by good agents
        rew = 0
        shaped_reward = True
        good_agents = self.good_agents(world)
        
        # Positive reward for reaching landmarks
        for landmark in world.landmarks:
            if self.is_collision(agent, landmark):
                rew += 5  # Reduced positive reward for reaching a landmark

        # Negative reward if caught by good agents
        for good_agent in good_agents:
            if self.is_collision(agent, good_agent):
                rew -= 20  # Increased negative reward for being caught by a good agent

        # Optionally, shape the reward based on distance to the nearest landmark
        if shaped_reward:
            min_distance = min(
                np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
                for landmark in world.landmarks
            )
            rew -= 0.05 * min_distance  # Reduced negative reward proportional to the distance to the nearest landmark

        return rew
    """
    """
    def agent_reward(self, agent, world):
        # Good agents are rewarded for catching adversaries and penalized if adversaries reach landmarks
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        good_agents = self.good_agents(world)
        
        # Optional shaping reward based on distance to adversaries
        if shape:
            for adv in adversaries:
                rew += 0.1 * min(
                    np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
                    for agent in good_agents
                )
        
        # Positive reward for catching adversaries
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew += 10
        
        # Negative reward if adversaries reach landmarks
        for adv in adversaries:
            for landmark in world.landmarks:
                if self.is_collision(adv, landmark):
                    rew -= 10

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for reaching landmarks and negatively rewarded if caught by good agents
        rew = 0
        shaped_reward = True
        good_agents = self.good_agents(world)
        
        # Positive reward for reaching landmarks
        for landmark in world.landmarks:
            if self.is_collision(agent, landmark):
                rew += 10  # Increased positive reward for reaching a landmark

        # Negative reward if caught by good agents
        for good_agent in good_agents:
            if self.is_collision(agent, good_agent):
                rew -= 5  # Reduced negative reward for being caught by a good agent

        # Optionally, shape the reward based on distance to the nearest landmark
        if shaped_reward:
            min_distance = min(
                np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
                for landmark in world.landmarks
            )
            rew -= 0.1 * min_distance  # Reduced negative reward proportional to the distance to the nearest landmark

        return rew
    """
    """
    def agent_reward(self, agent, world):
        # Good agents are rewarded for catching adversaries and penalized if adversaries reach landmarks
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        good_agents = self.good_agents(world)
        
        # Optional shaping reward based on distance to adversaries
        if shape:
            for adv in adversaries:
                rew -= 0.1 * min(
                    np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
                    for agent in good_agents
                )
        
        # Positive reward for catching adversaries
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew += 10
        
        # Negative reward if adversaries reach landmarks
        for adv in adversaries:
            for landmark in world.landmarks:
                if self.is_collision(adv, landmark):
                    rew -= 10
        
        # Penalty for going out of bounds
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        
        return rew
    
    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for reaching landmarks and negatively rewarded if caught by good agents
        rew = 0
        shaped_reward = True
        good_agents = self.good_agents(world)
        
        # Positive reward for reaching landmarks
        for landmark in world.landmarks:
            if self.is_collision(agent, landmark):
                rew += 5  # Positive reward for reaching a landmark

        # Negative reward if caught by good agents
        for good_agent in good_agents:
            if self.is_collision(agent, good_agent):
                rew -= 10  # Negative reward for being caught by a good agent

        # Optionally, shape the reward based on distance to the nearest landmark
        if shaped_reward:
            min_distance = min(
                np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
                for landmark in world.landmarks
            )
            rew -= min_distance  # Negative reward proportional to the distance to the nearest landmark

        # Penalty for going out of bounds
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew
    """
    def observation(self, agent, world):
        # Get positions of all landmarks in this agent's reference frame
        landmark_pos = []
        for landmark in world.landmarks:
            landmark_pos.append(landmark.state.p_pos - agent.state.p_pos)
        
        # Get positions of all obstacles in this agent's reference frame
        obstacle_pos = []
        for obstacle in world.obstacles:
            obstacle_pos.append(obstacle.state.p_pos - agent.state.p_pos)
        
        # Communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        
        # Concatenate all observations
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.goal_a.state.p_pos - agent.state.p_pos]
            + landmark_pos
            + obstacle_pos
            + other_pos
            + other_vel
        )
