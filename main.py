import os
import gym
import time
import random
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from distutils.util import strtobool
#from base import base
#from agent import Agent
from pickletools import optimize
from torch.utils.tensorboard.writer import SummaryWriter






class base:
    def gym_env(gym_id, seed, idx, capture_video, run_name):
            def thunk():
                env = gym.make(gym_id)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                if capture_video:
                    if idx == 0:
                        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", record_video_trigger=lambda t: t % 1000 == 0)
                env.seed(seed)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env
            return thunk


    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"), 
                            help='the name of this experiment')
        parser.add_argument('--gym-id', type=str, default="CartPole-v1", 
                            help='the id of the gym environment')
        parser.add_argument('--learning-rate', type=float, default=2.5e-4, 
                            help='the learning rate of the optimizer')
        parser.add_argument('--seed', type=int, default=1, 
                            help='seed of the experiment')
        parser.add_argument('--total-timesteps', type=int, default=25000, 
                            help='total timesteps of the experiments')
        parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                            help="if toggled, `torch.backends.cudnn.deterministic=False`")
        parser.add_argument("--mps", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                            help="if toggled, mps will be enabled by default")
        parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                            help="if toggled, cuda will be enabled by default")
        parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                            help="if toggled, this experiment will be tracked with Weights and Biases")
        parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
                            help="the wandb's project name")
        parser.add_argument("--wandb-entity", type=str, default=None,
                            help="the entity (team) of wandb's project")
        parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                            help="whether to capture videos of the agent performances")
        
        parser.add_argument("--num-envs", type=int, default=4,
                            help="the number of parallel environment")
        parser.add_argument("--num-steps", type=int, default=128,
                            help="the number of steps to run in each environment per policy rollout")
        args = parser.parse_args()
        return args
    


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        
        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)
            return layer
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1), std=1.1)
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,envs.single_action_space), std=0.01)
        )
    
    def get_value(self, x):
        return self.critic(x)


if __name__ == '__main__':
    args = base.parse_args()
    run_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters", 
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.mps.deterministic = args.torch_deterministic

    device = torch.device("mps" if torch.cuda.is_available() and args.cuda else "cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "mps")
       
    envs = gym.vector.SyncVectorEnv([base.gym_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    print("envs.single_observation_space.shape", envs.single_observation_space.shape)
    print("envs.single_action_space.n", envs.single_action_space.n)
    
    
    agent = Agent(envs)#.to(device)
    print(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=args.eps)

    '''
    obs =torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    action =torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs =torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards =torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones =torch.zeros((args.num_steps, args.num_envs)).to(device)
    values =torch.zeros((args.num_steps, args.num_envs)).to(device)
                        
                        
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    print(num_updates)
    print("next_obs.shape", next_obs.shape)
    print("agent.get_value(next_obs)", agent.get_value(next_obs))
    print("agent.get_value(next_obs).shape", agent.get_value(next_obs).shape)
    '''