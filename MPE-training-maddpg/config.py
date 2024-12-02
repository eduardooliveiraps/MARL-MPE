import torch
from torch import multiprocessing

class Config:
    def __init__(self, env_name):
        self.env_name = env_name  # Store the environment name
        # Seed
        self.seed = 0
        torch.manual_seed(self.seed)
        # Devices
        is_fork = multiprocessing.get_start_method() == "fork"
        self.device = (
            torch.device(0)
            if torch.cuda.is_available() and not is_fork
            else torch.device("cpu")
        )
        # Sampling
        self.frames_per_batch = 1_000 # Number of team frames collected per sampling iteration
        self.n_iters = 10 # Number of sampling and training iterations
        self.total_frames = self.frames_per_batch * self.n_iters
        # Replay buffer
        self.memory_size = 1_000_000 # The replay buffer of each group can store this many frames
        # Training
        self.n_optimiser_steps = 100 # Number of optimization steps per training iteration
        self.train_batch_size = 128 # Number of frames trained in each optimiser step
        self.lr = 3e-4 # Learning rate
        self.max_grad_norm = 1.0 # Maximum norm for the gradients
        # DDPG
        self.gamma = 0.99 # Discount factor
        self.polyak_tau = 0.005 # Tau for the soft-update of the target network
        # Define the environment
        self.max_steps = 100 # Environment steps before done
        self.n_chasers = 2
        self.n_evaders = 1
        self.n_obstacles = 2
        self.use_vmas = False # Set this to True for a great performance speedup

        # Environment-specific configurations
        if env_name in ["simple_tag", "custom_environment"]:
            self.iteration_when_stop_training_evaders = self.n_iters // 2
        elif env_name in ["simple_reference", "simple_crypto"]:
            self.train_batch_size = 10