import os
import gym
import argparse

from distutils.util import strtobool


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