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

