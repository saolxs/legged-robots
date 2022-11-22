import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter



class loader:
    
    def parse_args():
        # fmt: off
        parser = argparse.ArgumentParser()
        parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
            help="the name of this experiment")
        parser.add_argument("--gym-id", type=str, default="CartPole-v1",
            help="the id of the gym environment")
        parser.add_argument("--learning-rate", type=float, default=1e-4,
            help="the learning rate of the optimizer")
        parser.add_argument("--seed", type=int, default=42,
            help="seed of the experiment")
        parser.add_argument("--total-timesteps", type=int, default=25000,
            help="total timesteps of the experiments")
        parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="if toggled, `torch.backends.cudnn.deterministic=False`")
        parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="if toggled, cuda will be enabled by default")
        parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
            help="if toggled, this experiment will be tracked with Weights and Biases")
        parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
            help="the wandb's project name")
        parser.add_argument("--wandb-entity", type=str, default=None,
            help="the entity (team) of wandb's project")
        parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="weather to capture videos of the agent performances (check out `videos` folder)")

        # Algorithm specific arguments
        parser.add_argument("--num-envs", type=int, default=4,
            help="the number of parallel game environments")
        parser.add_argument("--num-steps", type=int, default=128,
            help="the number of steps to run in each environment per policy rollout")
        parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="Toggle learning rate annealing for policy and value networks")
        parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="Use GAE for advantage computation")
        parser.add_argument("--gamma", type=float, default=0.99,
            help="the discount factor gamma")
        parser.add_argument("--gae-lambda", type=float, default=0.95,
            help="the lambda for the general advantage estimation")
        parser.add_argument("--num-minibatches", type=int, default=4,
            help="the number of mini-batches")
        parser.add_argument("--update-epochs", type=int, default=4,
            help="the K epochs to update the policy")
        parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="Toggles advantages normalization")
        parser.add_argument("--clip-coef", type=float, default=0.2,
            help="the surrogate clipping coefficient")
        parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
        parser.add_argument("--ent-coef", type=float, default=0.01,
            help="coefficient of the entropy")
        parser.add_argument("--vf-coef", type=float, default=0.5,
            help="coefficient of the value function")
        parser.add_argument("--max-grad-norm", type=float, default=0.5,
            help="the maximum norm for the gradient clipping")
        parser.add_argument("--target-kl", type=float, default=None,
            help="the target KL divergence threshold")
        
        args = parser.parse_args()
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        # fmt: on
        return args

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

class ReplayBuffer:
    def __init__(self, envs):
        device = loader.device
        args = loader.parse_args()
        # Storage setup
        self.state = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        
    def get_batches(self):
    # Optimizing the policy and value network
        indices = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(indices)
            batch_inds = [indices[i:i + args.minibatch_size] for i in range(0, args.batch_size, args.minibatch_size)]
                
            return batch_inds
                        
                                   
        
class Agent(nn.Module):
    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer 
    
    def __init__(self, envs, chkpt_dir='tmp/ppo'):
        super().__init__()
        
        self.checkpoint = os.path.join(chkpt_dir, 'actor_critic_ppo')
        
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.ReLU(),
            self.layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.ReLU(),
            self.layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            self.layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def save(self):
        torch.save(self.state_dict(), self.checkpoint)
    
    def load(self):
        torch.load_state_dict(torch.load(self.checkpoint))
        

class LearnPolicy:
    #generalized advantage estimate
    def __init__(self):
        pass
    
    def gae(self):
        with torch.no_grad():

            nxt_value = agent.get_value(nxt_state).reshape(1, -1)
            
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                advantage = 0
                
                for step in reversed(range(args.num_steps)):
                    if step == args.num_steps - 1:
                        mask_terminal  = 1.0 - nxt_done
                        nextvalues = nxt_value
                    else:
                        mask_terminal = 1.0 - dones[step + 1]
                        nextvalues = values[step + 1]
                        
                    delta = rewards[step] + args.gamma * nextvalues * mask_terminal - values[step]
                    advantages[step] = advantage = delta + args.gamma * args.gae_lambda * mask_terminal
                returns = advantages + values
                
            else:
                returns = torch.zeros_like(rewards).to(device)
                for step in reversed(range(args.num_steps)):
                    if step == args.num_steps - 1:
                        mask_terminal= 1.0 - nxt_done
                        nxt_return = nxt_value
                    else:
                        mask_terminal = 1.0 - dones[step + 1]
                        nxt_return = returns[step + 1]
                    returns[step] = rewards[step] + args.gamma * mask_terminal * nxt_return
                advantages = returns - values
                
            return advantages, returns      
    
    def normalize(self, x):
        x -= x.mean()
        x /= (x.std() + 1e-8)
        return x
    
    def mini_batch(self):
        


#class TestPolicy:
    
    
    
if __name__ == "__main__":
        
    args = loader.parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

   

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    device = loader.device
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    nxt_state = torch.Tensor(envs.reset()).to(device)
    nxt_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    
    mem = ReplayBuffer(envs)
    
    state, logprobs, values,\
    actions, rewards, dones,\
    batches  = mem.state, mem.logprobs, mem.values,\
                mem.actions, mem.rewards, mem.dones,\
                mem.get_batches()

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            state[step] = nxt_state
            dones[step] = nxt_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_value(nxt_state)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            nxt_state, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            nxt_state, nxt_done = torch.Tensor(nxt_state).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        advantages, returns = LearnPolicy.gae()
        # flatten the batch
        batch_obs = state.reshape((-1,) + envs.single_observation_space.shape)
        batch_logprobs = logprobs.reshape(-1)
        batch_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)

        for epoch in range(args.update_epochs):
            
            for i in range(0, args.batch_size, args.minibatch_size):
                    clipfracs = []
                    batch_inds = mem.get_batches()

                    _, newlogprob, entropy, newvalue = agent.get_action_value(b_obs[batch_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[batch_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[batch_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()