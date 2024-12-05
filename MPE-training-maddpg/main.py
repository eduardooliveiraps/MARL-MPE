import sys
import os
import copy
import torch
from tqdm import tqdm
import pygame
import calendar
import time
import tempfile
from torchrl.record import CSVLogger, PixelRenderTransform, VideoRecorder
from torchrl.envs import TransformedEnv, set_exploration_type, ExplorationType
from config import Config
from environment import create_env
from policy import create_policies, create_exploration_policies
from critic import create_critics
from collector import create_collector
from replay_buffer import create_replay_buffers
from loss import create_losses, create_optimizers, create_target_updaters
from training import process_batch, train
    

def list_models(diretorio):
    # Verify if the folder exist
    if not os.path.exists(diretorio):
        print(f"The folder '{diretorio}' does not exist.")
        return None

    # Listar files with the extension
    arquivos = os.listdir(diretorio)
    modelos = [arquivo for arquivo in arquivos if arquivo.endswith(('.pth'))]

    if not modelos:
        print("No models were found.")
        return None

    # List every model avaliable in the folder
    print("Trained models available:")
    for i, modelo in enumerate(modelos, start=1):
        print(f"{i}. {modelo}")

    # Choose the model
    while True:
        try:
            escolha = int(input("Enter the model number you wish to select: "))
            if 1 <= escolha <= len(modelos):
                modelo_escolhido = modelos[escolha - 1]
                print(f"You chose: {modelo_escolhido}")
                print("Press enter to continue...")
                input()
                return modelo_escolhido
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def process_input(input):
    if len(input) < 2:
        print("Usage: python main.py <environment_name>")
        sys.exit(1)
    if(input[1] == 'help'):
        help()
        sys.exit(1)
    #'MADDPG','IDDPG'
    env_names = [
        'custom_environment_v0',
        'simple_adversary_v3',
        'simple_crypto_v3',
        'simple_push_v3',
        'simple_reference_v3',
        'simple_speaker_listener_v4',
        'simple_spread_v3',
        'simple_tag_v3',
        'simple_v3',
        'simple_world_comm_v3'
        ]
    algorithms_list = [
        'MADDPG',
        'IDDPG',
    ]
    simple_envs = [env for env in env_names if input[1] in env]
    if(len(simple_envs) == 0):
        print("<environment_name> doesn't exist.It must be among this:")
        for env in env_names:
            print(env)
        sys.exit(1)
    else:
        env_name = simple_envs[0]

    steps = 100
    render = None
    alg = "MADDPG"
    vmas = False
    i=2
    while(i !=len(input)):
        if(input[i]=='-steps'):
            try:
                steps = int(input[i+1])
            except ValueError:
                print(f"After steps must have an integer number")
                sys.exit(1)
            steps = input[i+1]
            i = i+2
        elif((input[i]=='-render')):
            print(f"--Warning: render is only supported without vmas")
            render = True
            i = i+1
        elif((input[i]=='-alg')):
            aux = [algs for algs in algorithms_list if input[i+1].upper() in algs]
            if(len(aux) != 0):
                alg = aux[0]
            else:
                print(f"The algorithm is not supported, please choose something among this:")
                for aux1 in algorithms_list:
                    print(aux1)
                sys.exit(1)
            i = i+2
        elif((input[i]=='-vmas')):
            vmas = True
            i = i+1

    return env_name, steps, render, alg, vmas

def run_env(env_name, steps, render, alg, vmas, policy_flag):
    os.system('cls' if os.name == 'nt' else 'clear')
    config = Config(env_name,steps,alg,vmas)
    if(render==True):
        env = create_env(config,render_mode='human')
    else:
        env = create_env(config)
    data = env.reset()
    if(policy_flag==True and vmas==False):
        model_trained_name = list_models(f'./models/{env_name}')
        trained_policy = torch.load(f'./models/{env_name}/{model_trained_name}')
        env.rollout(int(steps),trained_policy)
    elif(policy_flag==False and vmas==False):       
        env.rollout(int(steps))
    elif(policy_flag==True and vmas==True):
        model_trained_name = list_models(f'./models/{env_name}')
        trained_policy = torch.load(f'./models/{env_name}/{model_trained_name}')
        render_vmas(env,steps,trained_policy=trained_policy)
    elif(policy_flag==False and vmas==True):
        render_vmas(env,steps)

    env.close()


def render_vmas(env,steps,trained_policy = None):
    with tempfile.TemporaryDirectory() as tmpdir:
        video_logger = CSVLogger("vmas_logs", tmpdir, video_format="mp4")
        print("Creating rendering env")
        env_with_render = TransformedEnv(env.base_env, env.transform.clone())
        env_with_render = env_with_render.append_transform(
            PixelRenderTransform(
                out_keys=["pixels"],
                # the np.ndarray has a negative stride and needs to be copied before being cast to a tensor
                preproc=lambda x: x.copy(),
                as_non_tensor=True,
                # asking for array rather than on-screen rendering
                mode="rgb_array",
            )
        )
        env_with_render = env_with_render.append_transform(
            VideoRecorder(logger=video_logger, tag="vmas_rendered")
        )
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            print("Rendering rollout...")
            if(trained_policy != None):
                env_with_render.rollout(steps,policy=trained_policy)
            else:
                env_with_render.rollout(steps)
        print("Saving the video...")
        env_with_render.transform.dump()
        print("Saved! Saved directory tree:")
        video_logger.print_log_dir()

def train_env(env_name, steps, render, alg, vmas, policy_flag):
    os.system('cls' if os.name == 'nt' else 'clear')
    config = Config(env_name,steps,alg,vmas)
    if(render==True):
        env = create_env(config,render_mode='human')
    else:
        env = create_env(config)
    
    policies, policy_modules = create_policies(env, config)
    exploration_policies = create_exploration_policies(policies, env, config)
    
    critics = create_critics(env, config)
    
    reset_td = env.reset(seed=42)
    
    '''
    for group, _agents in env.group_map.items():
        print(
            f"Running value and policy for group '{group}':",
            critics[group](policies[group](reset_td)),
        )
    '''

    collector = create_collector(env, exploration_policies, config)
    replay_buffers = create_replay_buffers(env, config)
    losses = create_losses(policies, critics, env, config)
    optimizers = create_optimizers(losses, config)
    target_updaters = create_target_updaters(losses, config)
    if(policy_flag==True):
        model_trained_name = list_models(f'./models/{env_name}')
        trained_policy = torch.load(f'./models/{env_name}/{model_trained_name}')
        collector.policy = trained_policy
    train(env, collector, replay_buffers, losses, optimizers, target_updaters, exploration_policies, config)


def help():
    print(
        'Welcome to MPE-training-maddpg\nThe programe can receive:\n<environment> (mandatory)\n\
        This is the first argument and its the only one who need to be in order\n\
        example:\n\
        \
        python main.py simple_tag\n\
        The program can be trained with this environments:\n\
        \
        - custom_environment_v0\n\
        \
        - simple_adversary_v3\n\
        \
        - simple_crypto_v3\n\
        \
        - simple_push_v3\n\
        \
        - simple_reference_v3\n\
        \
        - simple_speaker_listener_v4\n\
        \
        - simple_spread_v3\n\
        \
        - simple_tag_v3\n\
        \
        - simple_v3\n\
        \
        - simple_world_comm_v3\n\
        You can read more about them here:\n\
        https://pettingzoo.farama.org/environments/mpe/\n<steps>\n\
        This is the number of steps you want to execute, by default it is 100, which means you don\'t need to pass this argument if you don\'t want to.\n\
        To pass the argument, you need to write -steps <number of steps>\n\
        example:\n\
        \
        python main.py simple_tag -steps 100\n<alg>\n\
        The program can be trained with this algorithms:\n\
        \
        - MADDPG (Multi Agent Deep Deterministic Policy Gradient)\n\
        \
        - IDDPG (Independent Deep Deterministic Policy Gradient)\n\
        To pass the argument, you need to write -alg <acronym of the name>, by default it uses MADDPG so you don\'t need to pass the argument\n\
        example:\n\
        \
        python main.py simple_tag -alg MADDPG\n<render>\n\
        If you want to render and see what\'s happen with your environment you need to pass -render\n\
        example:\n\
        \
        python main.py simple_tag -steps 100 -alg maddpg -render\n<vmas>\n\
        If you want to use vmas insted of pettingzoo you need to pass -vmas\n\
        example:\n\
        \
        python main.py simple_tag -steps 100 -alg maddpg -render -vmas\n*Note: As mentioned the arguments don\'t need to be passed in order with the exception of env'
    )
    
def main(env_name, steps, render, alg, vmas):
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\nMenu:")
        print("1. Run the environment with training")
        print("2. Run the environment without training")
        print("3. Train the environment")
        print("4. Retrain the environment")
        print("0. Sair")

        try:
            escolha = int(input("choose an option: "))
        except ValueError:
            print("Please, insert a valid number")
            print("Press enter to continue...")
            input()
            continue

        if escolha == 0:
            print("Exiting the program...")
            sys.exit(1)
        elif escolha == 1:
            run_env(env_name, steps, render, alg, vmas, True)
            #visualizar_env(com_policy=True)
        elif escolha == 2:
            run_env(env_name, steps, render, alg, vmas, False)
            #visualizar_env(com_policy=False)
        elif escolha == 3:
            train_env(env_name, steps, render, alg, vmas, False)
            #treinar_env()
        elif escolha == 4:
            train_env(env_name, steps, render, alg, vmas, True)
            #retreinar_env()

        else:
            print("Invalid option, please try again.")
            print("Press enter to continue...")
            input()

if __name__ == "__main__":
    env_name, steps, render, alg, vmas = process_input(sys.argv)
    print("Press enter to continue...")
    input()
    main(env_name, steps, render, alg, vmas)