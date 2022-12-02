import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import pybullet_envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from actor_critic import Agent


def parse_args():
    # fmt: off

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
            help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="MinitaurBulletEnv-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
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
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
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


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #clip the continous action within the valid boundaries
        env = gym.wrappers.ClipAction(env)
        #create a utility class to keep track of the running means and std of the observations
        env = gym.wrappers.NormalizeObservation(env)
        #clip the normalized observation_space between -10 and 10
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        #calculate the discounted return and return the normalized return
        env = gym.wrappers.NormalizeReward(env)
        #clip the normalized reward between -10 and 10
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def compute_gae(agent, nxt_state, values, dones, nxt_done, rewards, args): 
        # bootstrap value if not done
        
        with torch.no_grad():
            nxt_value = agent.value(nxt_state).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                gae = 0
                for step in reversed(range(args.num_steps)):
                    if step == args.num_steps - 1:
                        mask_terminal  = 1.0 - nxt_done
                        nextvalues = nxt_value
                    else:
                        mask_terminal = 1.0 - dones[step + 1]
                        nextvalues = values[step + 1]
                        
                    delta = rewards[step] + args.gamma * nextvalues * mask_terminal - values[step]
                    
                    advantages[step] = gae = delta + args.gamma * args.gae_lambda * mask_terminal * gae
                    
                returns = advantages + values
                
            else:
                returns = torch.zeros_like(rewards).to(device)
                advantages = []
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


def get_batches(envs, args, optimizer, agent, device):
    
    #capture datapoints
    state = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    #track the number of environment steps
    global_step = 0
    start_time = time.time()
    nxt_state = torch.Tensor(envs.reset()).to(device)
    nxt_done = torch.zeros(args.num_envs).to(device)
    #calculate the number of updates for the training
    num_updates = args.total_timesteps // args.batch_size
    

   
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
                action, logprob, _, value = agent.action_value(nxt_state)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            nxt_state, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            nxt_state, nxt_done = torch.Tensor(nxt_state).to(device), torch.Tensor(done).to(device)
            
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, return_per_episode={item['episode']['r']}")
                    writer.add_scalar("charts/return_per_episode", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/length_per_episode", item["episode"]["l"], global_step)
                    break
                
        advantages, returns = compute_gae(agent, nxt_state, values, dones, nxt_done, rewards, args)
                
        yield state.reshape((-1,) + envs.single_observation_space.shape), logprobs.reshape(-1), actions.reshape((-1,) + envs.single_action_space.shape), advantages.reshape(-1), returns.reshape(-1), values.reshape(-1), global_step
        


def update_ppo(envs, args, agent):
    
    start_time = time.time()
    
    for batch_states, batch_logprobs, batch_actions, batch_advantages, batch_returns, batch_values, global_step in get_batches(envs, args, optimizer, agent, device):

        # Optimizing the policy and value network
        fracs_clip= []
        batch_inds = np.arange(args.batch_size)
        
        for epoch in range(args.update_epochs):

            np.random.shuffle(batch_inds)
        
            for start in range(0, args.batch_size, args.minibatch_size):
                
                end = start + args.minibatch_size
                minibatch_inds = batch_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.action_value(batch_states[minibatch_inds], batch_actions[minibatch_inds])
                log_ratio = newlogprob - batch_logprobs[minibatch_inds]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    fracs_clip += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                minibatch_advantages = batch_advantages[minibatch_inds]
                if args.norm_adv:
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -minibatch_advantages * ratio
                pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - batch_returns[minibatch_inds]) ** 2
                    v_clipped = batch_values[minibatch_inds] + torch.clamp(
                        newvalue - batch_values[minibatch_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - batch_returns[minibatch_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - batch_returns[minibatch_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
                
        y_pred, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        #writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/frac_clip", np.mean(fracs_clip), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("Time Taken:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/time", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

      
if __name__ == "__main__":
        
    args = parse_args()
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

    #seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    print(envs.single_observation_space)
    

    assert isinstance(envs.single_action_space, gym.spaces.Box)
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    #capture datapoints
    state = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    #track the number of environment steps
    global_step = 0
    start_time = time.time()
    nxt_state = torch.Tensor(envs.reset()).to(device)
    nxt_done = torch.zeros(args.num_envs).to(device)
    #calculate the number of updates for the training
    num_updates = args.total_timesteps // args.batch_size


    #get_batches(envs, args, optimizer, agent, device)
    
    update_ppo(envs, args, agent)