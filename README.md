# Robonuts_Final_Project

This is the final project repository for Robonuts Team (G12) for the course "Intro to application programming for engineers".
Team members: Paria Esmatloo, Elena Soto

Please navigate inside the repository to review the script and results for implementing two reinforcement learning algorithms on Reacher-v2 environment from OpenAI Gym.

This repository includes implementation of REINFORCE method as well as Actor-Critic method. A random agent implementation is also included as a baseline comparison.


Instructions to set up the requirements
- Install Gym (https://gym.openai.com/docs/)
- In order to simulate physics based environments, you have to install and activate Mujoco. You can request a 30-day trial license or a 1-year student license from here: https://www.roboti.us/license.html
- We have prepared a guide that explains the steps for Mujoco-Py installation for both MAC users and linux users: https://docs.google.com/document/d/1Uq5_59Kk4QHMNyWTU4aNXyZWj4zSrBBZePHKIq5SIxk/edit
- In order to run the Actor-Critic code, you should install Pytorch: https://pytorch.org/
- The REINFORCE code uses Tensorflow agents. Tensorflow installation instructions : https://www.tensorflow.org/install
- Please not that on ubuntu, you may need to have compatibility issues getting REINFORCE to run. Mujoco-py requires numpy version 20, but Tensorflow 2 supports up to an earlier version.

# Actor Critic Method
Once you have all the dependencies installed, download "actor_critic_continuous.py", and "main_reacher_ac_mean.py" and make sure they are in the same directory. Run main_reacher_ac.mean.py. This program trains an Actor-Critic agent over 10 trials of 200 episodes. The outputs are CSV files containing episode number and total reward value for each episode in each trial, and figures demonstrating that. Moreover, averaged values over these 10 trials are also calculated and saved in a separate CSV file containing episode number, average, and standard deviation columns. The averaged values are also plotted in a separate figure similar to the one below.



![image](https://user-images.githubusercontent.com/77804192/117373694-5e44f300-ae91-11eb-967e-ed9cc72f09dc.png)
