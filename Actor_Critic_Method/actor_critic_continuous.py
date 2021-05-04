
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 02:17:26 2021

@author: paria
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GenericNetwork(nn.Module):
    def __init__(self,lr,input_dims,fc1_dims,fc2_dims,n_actions):
        super(GenericNetwork,self).__init__()
        self.lr=lr
        self.input_dims=input_dims
        self.fc1_dims= fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(),lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self,observation):
        state = T.tensor(observation , dtype = T.float).to(self.device) 

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
class Agent(object):

    
    def __init__(self,alpha,beta,input_dims,gamma=0.99,n_actions=4,  
                 layer1_size=64, layer2_size=64, n_outputs=1):
        self.gamma = gamma
        self.log_probs = None
        self.log_probs1 = None
        self.log_probs2 = None

        self.n_outputs = n_outputs
        self.actor = GenericNetwork(alpha, input_dims, layer1_size, 
                                    layer2_size, n_actions=n_actions)
        
        self.critic = GenericNetwork(beta, input_dims, layer1_size, 
                                     layer2_size, n_actions=1) 
        
    def choose_action(self, observation):
        mu1, sigma1, mu2, sigma2  = self.actor.forward(observation)
        sigma1 = T.exp(sigma1) #to make it positive
        sigma2 = T.exp(sigma2)
        
        action_probs1 = T.distributions.Normal(mu1,sigma1)  #agent learns mu and sigma to maximize reward
        action_probs2 = T.distributions.Normal(mu2,sigma2)  #agent learns mu and sigma to maximize reward

        probs1 = action_probs1.sample(sample_shape=T.Size([self.n_outputs]))
        probs2 = action_probs2.sample(sample_shape=T.Size([self.n_outputs]))

        self.log_probs1 = action_probs1.log_prob(probs1).to(self.actor.device)
        self.log_probs2 = action_probs2.log_prob(probs2).to(self.actor.device)
        
        action1= T.tanh(probs1) #number bw +-1
        action2 = T.tanh(probs2)
        
        return action1.item() , action2.item()
    
    def learn(self,state,reward, new_state, done):
        
       self.actor.optimizer.zero_grad()
       self.critic.optimizer.zero_grad()
       
       critic_value_ = self.critic.forward(new_state) # value for the old state
       critic_value = self.critic.forward(state) # value for new state
       
       reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
       
       delta=reward + self.gamma*critic_value_*(1-int(done)) - critic_value

       actor_loss = -self.log_probs1* delta  
       # actor_loss = -m.sqrt((self.log_probs1 * delta)**2 +(self.log_probs2 * delta)**2)  # should I change this to include 2 outputs? and use only 1 logprobs?
       
       critic_loss = delta**2
       
       (actor_loss + critic_loss).backward() # sum the two losses and backpropagate
       self.actor.optimizer.step()
       self.critic.optimizer.step()


       
        
        
        