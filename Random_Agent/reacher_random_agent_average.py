#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 01:30:36 2021

@author: paria
"""



import argparse

import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt
import csv
import torch as T

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make('Reacher-v2')

    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir , force=True) 
    # env = gym.wrappers.Monitor(env, "recording")
    # env.seed(0)
    agent = RandomAgent(env.action_space)

    num_episodes = 200
    reward = 0
    done = False


    score_history = []
    for trial in range(10):
        score_history.append([])
        params_filename = "Rand_" + str(num_episodes)+"_tr_" + str(trial)
        csv_filename = params_filename + ".csv"
        fig_filename = params_filename + ".png"
        with open (csv_filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            for i in range(num_episodes):
                ob = env.reset()
                step_count=0
                score =0
                
                while True:
                # while not done:
                    action = agent.act(ob, reward, done)
                    ob,reward,done, _ = env.step(action)
                    
                    step_count+=1
                    score+=reward
                    # print(i, ' : ', reward)
                    # env.render()   # uncomment if you want to see the simulation in mujoco
                    if done:
                        break                    
                print('trial: ', trial, ',episode: ',i, ',score: %.2f' %score)
                    
                score_history[trial].append(score)
                csvwriter.writerow([i,score])   
                 

                    # Note there's no env.render() here. But the environment still can open window and
                    # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                    # Video is not recorded every episode, see capped_cubic_video_schedule for details.
                # print('episode: ', i, ', avg reward: ', reward_sum/step_count)
        # Close the env and write monitor result info to disk
        
        fig, ax = plt.subplots()
        ax.plot(range(num_episodes), score_history[trial])
    
        ax.set(xlabel='Episode', ylabel='Total Score',
               title='Random Agent Results')
        ax.grid()
        
        fig.savefig(fig_filename)
        plt.show()        
        env.close()
        
        
    params_filename_average = "AVG_"+ "Rand_" + str(num_episodes)
    csv_filename_average = params_filename_average + ".csv"
    fig_filename_average = params_filename_average + ".png"    
    
    m_score=[]
    std_score =[]
    T_score = T.tensor(score_history)


    with open (csv_filename_average, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(num_episodes):    
    
            mean_score = T.mean(T_score[:,i].float()).item()
            m_score.append(mean_score)
            
            standard_score = T.std(T_score[:,i].float()).item()
            std_score.append(standard_score)
            
            csvwriter.writerow([i,mean_score, standard_score])
            
        
    fig, ax = plt.subplots()
    

    ax.plot(range(num_episodes), m_score)
    
    m_plus_std = []
    m_minus_std =[]
    for (mm , ss) in zip ( m_score, std_score):
        m_plus_std.append(mm+ss)
        m_minus_std.append(mm-ss)
    
    ax.fill_between(range(num_episodes),m_minus_std , m_plus_std , alpha=0.2)

    ax.set(xlabel='Episode', ylabel='Total Score',
           title='Averaged Random Agent Results')
    ax.grid()
    

    fig.savefig(fig_filename_average)
    plt.show()    
