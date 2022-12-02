# legged-robots
implementation of ppo on legged robots

# Code-Base-for-Learning-Locomotion-Skills-on-Legged-Robots
This repository contains the Code and Video Recordings for the Learning Locomotion Skills for legged robots Research Report by Samuel Oladejo.

# Instructions

In order to run and compile the PPO Agents follow the following steps. Note that version of gym required is 0.23.0.

1) Clone the GitHub repository,
2) Open the folder in your terminal or preferred IDE,
3) Run 'pip install -r requirements.txt' in terminal,
4) Change Directory to the preferred one. i.e 'cd [name]_ppo'
4) In the preferred directory, Run 'python3 [name]_ppo.py'.

- Note: If you face any errors while running the code, please refer to StackOverflow to fix dependency issues. Furthermore, you can adjust the parameters by runnning 'python3 [name]_ppo.py -- time-steps [num_of_timesteps] --learning-rate [preferred_learning_rate]' 

- name: 'cartpole' | 'halfcheetah' | 'ant' | 'minitaur' 
