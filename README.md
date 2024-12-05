# Multi-Agent Particle Environments - Mixed Cooperative-Competitive Environments

## Overview
Reinforcement learning (RL) has demonstrated remarkable success in solving challenging problems, from game playing to robotics, and is emerging as a key component in large-scale industrial systems. However, most successes have focused on single-agent domains, where modeling the behavior of other actors is unnecessary.

This project is inspired by the paper **[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)**, which adapts actor-critic methods to consider action policies of other agents. It presents:
- A learning approach that enables agents to develop complex multi-agent coordination strategies.
- An ensemble-based training regimen for agents, enhancing robustness and adaptability in multi-agent environments.

Using the **PettingZoo MPE environments** as a foundation, this project introduces a custom environment that builds on the principles of cooperative and competitive multi-agent RL. Additionally, it allows for the training of agents in Multi-Agent Particle Environments (MPE) using the MADDPG algorithm.

## References
1. **Paper**: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)  
2. **Official Code Repository**: [OpenAI Multiagent Particle Envs](https://github.com/openai/multiagent-particle-envs)
3. **PettingZoo MPE Documentation**: [MPE Environments](https://pettingzoo.farama.org/environments/mpe/)

## Installation

### Prerequisites
This project requires **Python 3.8+**.  

To install the necessary dependencies:
1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```
2. Install the required packages manually:
```bash
pip install pettingzoo
pip install pygame
pip install numpy
pip install gymnasium
pip install supersuit
pip install stable-baselines3
pip install torch
pip install torchrl
pip install vmas
pip install tqdm
```
Or, install the dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Usage

### Running Standard MPE Environments

The MPE environments are located in the `MPE-environments/envs` directory. Each environment can be run directly by executing its corresponding Python file:
```bash
python <environment-name>.py
```

### Running Custom MPE
To explore the custom environment:
1. Navigate to the `custom-environment` directory:
```bash
cd custom-environment
```
2. Run the custom environment:
```bash
python main.py
```

The custom environment's implementation is located in `custom-env/env/custom_environment.py`.

### Training Agents on MPE using MADDPG
To train agents on the multi-agent particle environments:
1. Navigate to the `MPE-training-maddpg` directory:
```bash
cd MPE-training-maddpg
```

2. Run the script 'main.py':
```bash
 python main.py <arguments>
```

### Script Arguments
The Script can receive 5 arguments (1 mandatory, 4 optional) the mandatory argument must be the 1st argument, the other 4 do not have a specific order.
#### Arguments:

1. environment-name (Mandatory)

This is the first argument and its the only one who need to be in order
example:
```bash
python main.py simple_tag
```
The program can be trained with this environments:<br>
   - custom_environment_v0<br>
   - simple_adversary_v3<br>
   - simple_crypto_v3<br>
   - simple_push_v3<br>
   - simple_reference_v3<br>
   - simple_speaker_listener_v4<br>
   - simple_spread_v3<br>
   - simple_tag_v3<br>
   - simple_v3<br>
   - simple_world_comm_v3<br>
   
You can read more about them here:<br>
https://pettingzoo.farama.org/environments/mpe/

2. steps
   
This is the number of steps you want to execute, by default it is 100, which means you don't need to pass this argument if you don't want to.
To pass the argument, you need to write -steps <number-of-steps>
example:
```bash
python main.py simple_tag -steps 100
```

3. alg
   
The program can be trained with this algorithms:
    - MADDPG (Multi Agent Deep Deterministic Policy Gradient)
    - IDDPG (Independent Deep Deterministic Policy Gradient)
To pass the argument, you need to write -alg <acronym-of-the-name>, by default it uses MADDPG so you don't need to pass the argument
example:
```bash
python main.py simple_tag -alg MADDPG
```

4. render

If you want to render and see what's happen with your environment you need to pass -render
This just work with Pettingzoo, if you pass the vmas this will not work.
example:
```bash
 python main.py simple_tag -steps 100 -alg maddpg -render
```

5. vmas
   
If you want to use vmas insted of pettingzoo you need to pass -vmas
example:
```bash
python main.py simple_tag -steps 100 -alg maddpg -render -vmas
```


In this work we chose this 3 environments (not counting our custom environment) because of their different characteristics:
- **Simple Tag**: Competitive environment where agents are rewarded for capturing the adversary, without communication.
- **Simple Reference**: Cooperative environment where agents are rewarded for reaching landmarks, with communication.
- **Simple Crypto**: Mixed cooperative-competitive environment where agents are rewarded for reaching landmarks, with communication.

## Script

After running the script you will be able to choose between 4 options:

- **Run the environment with training**: This option will run the environment that you choose with the training module that you choose too.
- **Run the environment without training**: This option will run the environment that you choose with a random policy (if you want to see how it looks the environment).
- **Train the environment**: This option will start the training with the environment you choose. If you do not pass, the number of steps will be 100, more than enough for some environments. The training is saved every 10% of the iterations.
- **Retrain the environment**: This option will retraining with the environment you choose.Perfect if, for some reason, you need to stop the training or if the steps that you choose wasn't enough for learning well. If you do not pass, the number of steps will be 100, more than enough for some environments. The training is saved every 10% of the iterations.

### Saved models

The models will be saved in the folder **models** and divided by **envs**. <br>
Example:<br>
<div align="center">
 <div align="left"> 
\\**models**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---> \\**simple_tag_v3**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--->**simple_tag_v3_1733433595_True__300.pth**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---> \\**custom_environment_v0**<br>
  </div>
</div>

The name of the model will follow this logic {**env name**}\_{**timestamp da data**}\_{**algorithm**}\_{**vmas if exist**}\_{**steps**}.pth 

Example:
**simple_tag_v3_1733433595_True__300.pth**

- env name = simple_tag_v3
- timestamp da data = 1733433595
- algorithm = True (This means that the algorithm we use is MADDPG otherwise will be false)
- vmas doens't exist so we dont have this in the name
- steps = 300


## Custom Environment

### Environment Description
The custom environment is a mixed cooperative-competitive environment where the good agents (bluish) are slower and are tasked with intercepting the adversary (reddish). The adversary is faster and is rewarded for reaching landmarks (greenish) without being detected by the agents. The environment includes obstacles (grayish) that block movement.
By default, there are 3 agents, 1 adversary, 3 landmarks, and 2 obstacles.

<p align="center">
    <img src="docs/images/custom_environment.png" alt="Custom Environment" title="Custom Environment" width="450">
</p>

### Real-world Applications
This environment can be used to model various real-world scenarios, such as:
- **Cybersecurity Defense**: Agents protect a network from malicious intruders while avoiding detection.
- **Wildlife Conservation**: Drones monitor protected areas and prevent poachers from crossing boundaries.
- **Military Strategy and Defense**: Systems work together to secure areas, stopping adversaries from infiltrating. 
