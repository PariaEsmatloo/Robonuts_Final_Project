
"""
Created on Sun May  2 16:10:59 2021

@author: paria
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gym
from actor_critic_continuous import Agent
import matplotlib.pyplot as plt
import csv
import statistics as s
import torch as T
import time
from datetime import timedelta


if __name__ == '__main__':
    
    start_time = time.time()

    alpha_ac=0.000005
    layer1_size_ac= layer2_size_ac = 256
    beta_ac= 0.00001
    input_dims_ac=[11]
    gamma_ac=0.99
    env = gym.make('Reacher-v2')
    score_history = []
    num_episodes =200
    num_trials = 10
        
    for trial in range(num_trials):
        agent = Agent(alpha=alpha_ac, beta= beta_ac, input_dims=input_dims_ac ,gamma=gamma_ac,
              layer1_size=layer1_size_ac, layer2_size=layer2_size_ac)
        score_history.append([])
        params_filename = "AC_" + str(num_episodes)+"_alpha_" + str(alpha_ac)+ "_layer_"+ \
            str(layer1_size_ac) + "_lp_" + str(1) + "_tr_" + str(trial)
        
        csv_filename = params_filename + ".csv"
        fig_filename = params_filename + ".png"
        
        with open (csv_filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            for i in range(num_episodes):
                done= False
                score=0
                observation = env.reset()
                while not done:
                    action = np.array(agent.choose_action(observation)).reshape((2,)) 
                    observation_, reward, done, infor = env.step(action)
                    env.render()
                    agent.learn(observation, reward, observation_, done)
                    observation = observation_
                    score+=reward
                score_history[trial].append(score)
                print('trial: ', trial, ',episode: ',i, ',score: %.2f' %score)
                csvwriter.writerow([i,score])
        env.close()         
        fig, ax = plt.subplots()
        ax.plot(range(num_episodes), score_history[trial])    
        ax.set(xlabel='Episode', ylabel='Total Score',
               title='Actor-Critic Results')
        ax.grid()
        fig.savefig(fig_filename)
        plt.show()

    params_filename_average = "AVG_"+ "AC_" + str(num_episodes)+"_alpha_" + str(alpha_ac)+ "_layer_"+ str(layer1_size_ac) + "_lp_" + str(2) + "_tr_" + str(trial)
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
           title='Averaged Actor-Critic Results')
    ax.grid()
    

    fig.savefig(fig_filename_average)
    plt.show()    
        
    end_time = time.time()
    elapsed = end_time-start_time
    
    print("elpsaed time: ", str(timedelta(seconds=elapsed)))    