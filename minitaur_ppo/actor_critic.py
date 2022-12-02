import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.optim as optim
from datetime import datetime
from torch.distributions.categorical import Categorical




#Neural Network
def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    #if isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer 
    
class Agent(nn.Module):
    
    def __init__(self, envs, chkpt_dir='tmp/ppo'):
        super().__init__()
        
        self.checkpoint = os.path.join(chkpt_dir, 'actor_critic_ppo')
        
        self.critic = nn.Sequential(
            init_layer(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.ReLU(),
            init_layer(nn.Linear(128, 64)),
            nn.ReLU(),
            init_layer(nn.Linear(64, 64)),
            nn.ReLU(),
            init_layer(nn.Linear(64, 1), std=1.0),
        )
        #actor neural network - generate mean for the normal distribution for each available action given the inputted state
        #the action weights are used to determine probabilities for each action (the higher the weight, the higher the probability)
        #uses ReLU activation functions for non-linearity, as well as a Sigmoid function at the final layer to ensure each weight can be used as a probability
        self.actor = nn.Sequential(
            init_layer(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.ReLU(),
            init_layer(nn.Linear(128, 64)),
            nn.ReLU(),
            init_layer(nn.Linear(64, 64)),
            nn.ReLU(),
            init_layer(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        #state independent log std
        self.log_std = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def value(self, x):
        return self.critic(x)

    def action_value(self, x, action=None):
        mean  = self.actor(x)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        probs = Normal(mean, std)
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def save(self):
        torch.save(self.state_dict(), self.checkpoint)
    
    def load(self):
        torch.load_state_dict(torch.load(self.checkpoint))
        
    '''  
    def save_weights(self, iteration, state_rms):
        torch.save({"current_policy_state_dict": self.current_policy.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                    "actor_scheduler_state_dict": self.actor_scheduler.state_dict(),
                    "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
                    "iteration": iteration,
                    "state_rms_mean": state_rms.mean,
                    "state_rms_var": state_rms.var,
                    "state_rms_count": state_rms.count}, self.env_name + "_weights.pth")

    def load_weights(self):
        checkpoint = torch.load(self.env_name + "_weights.pth")
        self.current_policy.load_state_dict(checkpoint["current_policy_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.actor_scheduler.load_state_dict(checkpoint["actor_scheduler_state_dict"])
        self.critic_scheduler.load_state_dict(checkpoint["critic_scheduler_state_dict"])
        iteration = checkpoint["iteration"]
        state_rms_mean = checkpoint["state_rms_mean"]
        state_rms_var = checkpoint["state_rms_var"]
    '''
