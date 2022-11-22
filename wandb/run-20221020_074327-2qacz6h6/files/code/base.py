import os
import gym
import time
import random
import torch
import argparse
import numpy as np

from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="weather to capture videos of the agent performances (check out `videos` folder)")
    
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    print(args)
    run_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            #monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters", 
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    ''''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.mps.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    
   envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, "videos", record_video_trigger=lambda t: t % 100 == 0)
    observation = env.reset()
    
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, done, info, _ = env.step(action)
        if done:
            observation = env.reset()
            print(f"episodic return: {info['episode']['r']}")
        env.close
        
        def gym_env(gym_id):
            def thunk():
                env = gym.make(gym_id)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env = gym.wrappers.RecordVideo(env, "videos", record_video_trigger=lambda t: t % 100 == 0)
                return env
            return thunk
        
        envs = gym.vector.SyncVectorEnv([gym_env(args.gym_id)])
    '''
    
    for i in range(100):
        writer.add_scalar("test_loss", i*2, global_step=i)