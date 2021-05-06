# Robonuts_Final_Project

This is the final project repository for Robonuts Team (G12) for the course "Intro to application programming for engineers".
Team members: Paria Esmatloo, Elena Soto

Please navigate inside the repository to review the script and results for implementing two reinforcement learning algorithms on Reacher-v2 environment from OpenAI Gym.

This repository includes implementation of REINFORCE method as well as Actor-Critic method. A random agent implementation is also included as a baseline comparison.

For more information and to learn more about the Reacher-v2 environment, please refer to our final presentation slides: https://docs.google.com/presentation/d/14hLAmN2Z3_ts44Ka-06PK0o7cxjKGGDEmc3-omIL80k/edit?usp=sharing

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

If you wish to render the environment, keep "env.render()" uncommented. But, if you want to increase the program speed, comment this line.


# REINFORCE Method

![image](https://user-images.githubusercontent.com/77804192/117374752-58e8a800-ae93-11eb-94a8-e8b2abb06669.png)

# Random Agent
The script "reacher_random_agent_average.py" under the Random_Agent directory tests a random agent over 10 trials of 200 episodes on the reacher environment. The outputs files are similar to the Actor-Critic method, and the averaged results are shown below:
![image](https://user-images.githubusercontent.com/77804192/117374089-212d3080-ae92-11eb-9e83-a3e21eda374a.png)


# Comparison of Results
The script "plot_results.py" under the main directory loads the averaged CSV files which are renamed and relocated to the main directory and visualizes all results in a single figure for easy comparison.
![image](https://user-images.githubusercontent.com/77804192/117374253-6ea99d80-ae92-11eb-93bd-c28ea3827cd5.png)


