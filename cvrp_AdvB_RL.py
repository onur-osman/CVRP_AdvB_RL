# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:40:29 2023

@author: TOPKAPI
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
import tensorflow as tf
import matplotlib.pyplot as plt
#from IPython import display
import keras.models as keras_models
#from keras.optimizers import Adam

class environment:
        
    #
    def __init__(self, dst, city_demands_new):
        self.dst = dst
        #self.v_capacity_new = v_capacity_new
        self.city_demands_new = city_demands_new
        #self.init_city = init_city
        #self.current_city = current_city
        self.full_filled_cities = []
        #self.observation_space = self.Observation_space(self)
        self.action_space = self.Action_space(self)
        
    class Action_space:
        def __init__(self, environment):
            self.environment = environment
            
        def sample(self):
            
            dst = self.environment.dst
            n_city = len(dst)
            return np.random.randint(n_city)
        
        
        def step(self, action, agent,agents):
            next_city = action
            #rew = -dst[self.environment.current_city, next_city]
            rew = -dst[agent.current_city, next_city]
            done = False

            if agent.v_capacity_new > 0:
                if next_city == agent.init_city or self.environment.city_demands_new[next_city] == 0:
                    rew = -5000
                    agent.total_reward += rew
                    done = True
                    agent.done = True
                else:
                    if self.environment.city_demands_new[next_city] <= agent.v_capacity_new:
                        agent.v_capacity_new = agent.v_capacity_new - self.environment.city_demands_new[next_city]
                        self.environment.city_demands_new[next_city]=0
                        self.environment.full_filled_cities.append(next_city)
                        agent.cities.append(next_city)
                        agent.total_reward += rew
                    else:
                        self.environment.city_demands_new[next_city]= self.environment.city_demands_new[next_city]-agent.v_capacity_new
                        agent.v_capacity_new = 0
                        self.environment.full_filled_cities.append(next_city)
                        agent.cities.append(next_city)
                        agent.total_reward += rew
            else:
                if next_city == agent.init_city:
                    agent.current_city = next_city
                    #rew += agent.total_reward
                    agent.total_reward += rew
                    rew += 1000
                    #agent.total_reward += rew
                    self.environment.full_filled_cities.append(next_city)
                    agent.cities.append(next_city)
                    done = True
                    for iagent in agents:
                        if iagent.current_city != iagent.init_city:
                            done = False
                            break
                    rew_plus = 0
                    #if done:
                        #for iagent in agents:
                        #    rew_plus += iagent.total_reward
                        #rew += n_agents*100 + rew_plus
                    #    agent.total_reward += 0
                    agent.done = True
                else:
                    #rew += agent.total_reward
                    #agent.total_reward += rew
                    rew = -5000
                    agent.total_reward += rew
                    done = True
                    agent.done = True
          
            agent.current_city = next_city

            return rew, done, self.environment.city_demands_new, self.environment.full_filled_cities
        
    def reset(self, city_demands,init_city, city_demands_random, init_city_random):
        if init_city_random:
            init_city = [np.random.randint(0, len(dst))] * n_agents
        else:
            init_city = [init_city] * n_agents

        if city_demands_random:
            r_min = 1
            r_max = 10
            city_demands = np.random.randint(r_min, r_max,(len(dst)))
            city_demands[init_city] = 0
            residu = round(np.sum(city_demands)/n_agents)*n_agents - np.sum(city_demands)
            v_capacity_default = [round(np.sum(city_demands)/n_agents)] * n_agents
            v_capacity_default[0] = v_capacity_default[0] - residu
        else:
            city_demands = city_demands
            residu = round(np.sum(city_demands)/n_agents)*n_agents - np.sum(city_demands)
            v_capacity_default = [round(np.sum(city_demands)/n_agents)] * n_agents
            v_capacity_default[0] = v_capacity_default[0] - residu
               
        self.city_demands_new = city_demands.copy()
        self.full_filled_cities = []
        return self.city_demands_new, v_capacity_default, init_city

class agent:
    def __init__(self, v_capacity_new, cities, init_city, current_city, total_reward):
        self.v_capacity_new = v_capacity_new
        self.cities = cities
        self.init_city = init_city
        self.current_city = current_city
        self.total_reward = total_reward
        self.done = False
        self.state = []
        self.q_s = []
        self.q_s_next = []
        self.action = []
        self.rew = []

    def step(self, total_reward, next_city, city_demands_new):
        self.total_reward += total_reward
        self.cities.appand(next_city)
        if city_demands_new[next_city] <= self.v_capacity_new:
            self.v_capacity_new = self.v_capacity_new - city_demands_new[next_city]
            #city_demands_new[next_city] = 0
        else:
            self.v_capacity_new = 0
            #city_demands_new[next_city] = city_demands_new[next_city] - self.v_capacity_new
        
        return self.total_reward, self.cities, self.v_capacity_new
    
    def info(self):
        return self.v_capacity_new, self.init_city, self.current_city
        
    def reset(self, v_capacity_new, init_city, current_city):
        self.v_capacity_new = v_capacity_new
        self.init_city = init_city
        self.current_city = current_city
        self.total_reward = 0
        self.cities = []
        self.done = False
        self.state = []
        self.q_s = []
        self.q_s_next = []
        self.action = []
        self.rew = []
        return self.v_capacity_new, self.current_city, self.total_reward, self.cities


def visualize_loss(loss_values):
  display.clear_output(wait=True)
  display.display(plt.gcf())
  plt.clf()
  plt.title('Training...')
  plt.xlabel('Number of Episodes')
  plt.ylabel('Loss Value')
  plt.plot(range(len(loss_values)), loss_values)
  #plt.ylim(ymin=0)
  plt.show(block=False)


def confidence_by_state(q_values):
    q_values = q_values / np.absolute(q_values).max()
    sft = softmax(q_values)
    ix = sft.argmax()
    nsft = np.delete(sft, ix, 1)
    return 1-(nsft.mean()/sft.max())


def softmax(x):
    x_norm = x - np.min(x)
    x_norm = x_norm / np.max(x_norm)
    e_x = np.exp(x_norm / np.max(x_norm)) - 1
    return e_x / e_x.sum(axis=-1)

##env yarat
'''
dst = np.array([
    [0,5,15,20,25],
    [5,0,10,40,45],
    [15,10,0,15,60],
    [20,40,15,0,20],
    [25,45,60,20,0]
])

dst = np.array([
    [ 0, 5,15,20,25,13,22,12,18,44],
    [ 5, 0,10,40,45,33,14,26,48,61],
    [15,10, 0,15,60,21,17,10,33,18],
    [20,40,15, 0,20, 15,22,33,5,42],
    [25,45,60,20, 0,32,41,24,17,28],
    [13,33,21,15,32, 0,14,42,61,24],
    [22,14,17,22,41,14, 0,18,20,32],
    [12,26,10,33,24,42,18, 0,24,12],
    [18,48,33, 5,17,61,20,24, 0,15],
    [44,61,18,42,28,24,32,12,15, 0]
])

dst = np.array([
    [ 0, 5,15,20,25,13,22,12,18,44,23,12,22,18,14],
    [ 5, 0,10,40,45,33,14,26,48,61,43,34,16,28,41],
    [15,10, 0,15,60,21,17,10,33,18,12,55,37,30,64],
    [20,40,15, 0,20, 15,22,33,5,42,42,33,52,61,24],
    [25,45,60,20, 0,32,41,24,17,28, 9,16,35,57,49],
    [13,33,21,15,32, 0,14,42,61,24,19,24,38,62,32],
    [22,14,17,22,41,14, 0,18,20,32,10,30,24,42,59],
    [12,26,10,33,24,42,18, 0,24,12,62,18,26,33,70],
    [18,48,33, 5,17,61,20,24, 0,15,42,24,35,18,20],
    [44,61,18,42,28,24,32,12,15, 0,33, 7,19,35,15],
    [23,43,12,42, 9,19,10,62,42,33, 0,16,43,17,57],
    [12,34,55,33,16,24,30,18,24, 7,16, 0,17,20,33],
    [22,16,37,52,35,38,24,26,35,19,43,17, 0,20,10],
    [18,28,30,61,57,62,42,33,18,35,17,20,20, 0,54],
    [14,41,64,24,49,32,59,70,20,15,57,33,10,54, 0]
])

dst=np.array([
[0,47,13,47,30,55,48,30,24,27,65,35,34,34,67,19,63,6,27,55],
[47,0,27,45,34,5,44,44,47,27,37,51,63,8,11,47,55,70,13,29],
[13,27,0,49,29,61,24,13,49,18,62,54,32,5,14,23,62,44,26,23],
[47,45,49,0,33,64,66,38,46,52,6,42,8,32,35,6,9,65,40,29],
[30,34,29,33,0,29,14,14,28,27,56,37,35,13,63,49,60,48,69,69],
[55,5,61,64,29,0,21,46,53,37,61,17,13,5,15,40,38,30,25,5],
[48,44,24,66,14,21,0,58,47,34,21,58,59,61,35,69,60,10,20,58],
[30,44,13,38,14,46,58,0,31,35,67,68,55,42,65,37,15,26,24,41],
[24,47,49,46,28,53,47,31,0,9,9,16,67,58,51,69,70,70,14,68],
[27,27,18,52,27,37,34,35,9,0,40,9,25,64,60,5,47,58,21,9],
[65,37,62,6,56,61,21,67,9,40,0,22,11,36,32,30,63,32,23,8],
[35,51,54,42,37,17,58,68,16,9,22,0,19,20,6,51,5,45,31,21],
[34,63,32,8,35,13,59,55,67,25,11,19,0,48,26,11,40,15,63,48],
[34,8,5,32,13,5,61,42,58,64,36,20,48,0,60,55,58,46,51,50],
[67,11,14,35,63,15,35,65,51,60,32,6,26,60,0,26,40,62,8,38],
[19,47,23,6,49,40,69,37,69,5,30,51,11,55,26,0,33,64,46,69],
[63,55,62,9,60,38,60,15,70,47,63,5,40,58,40,33,0,43,60,35],
[6,70,44,65,48,30,10,26,70,58,32,45,15,46,62,64,43,0,40,16],
[27,13,26,40,69,25,20,24,14,21,23,31,63,51,8,46,60,40,0,46],
[55,29,23,29,69,5,58,41,68,9,8,21,48,50,38,69,35,16,46,0]
])
'''
# distance matrix of the cities
dst=np.array([
[0,47,13,47,30,55,48,30,24,27,65,35,34,34,67,19,63,6,27,55, 31, 54, 37, 30, 9, 19, 40, 32, 64, 8, 34, 40, 13, 40, 61, 18, 15, 9, 48, 6, 24, 69, 55, 21, 50, 14, 46, 61, 64, 27],
[47,0,27,45,34,5,44,44,47,27,37,51,63,8,11,47,55,70,13,29, 44, 35, 24, 13, 24, 15, 25, 12, 59, 69, 19, 51, 7, 45, 49, 7, 5, 14, 61, 23, 40, 39, 42, 26, 32, 32, 69, 8, 31, 57],
[13,27,0,49,29,61,24,13,49,18,62,54,32,5,14,23,62,44,26,23, 11, 67, 5, 24, 8, 34, 57, 65, 40, 58, 42, 54, 5, 36, 21, 25, 68, 18, 39, 19, 42, 70, 13, 68, 13, 17, 47, 13, 10, 48],
[47,45,49,0,33,64,66,38,46,52,6,42,8,32,35,6,9,65,40,29, 23, 51, 48, 32, 37, 67, 67, 53, 30, 7, 43, 42, 28, 63, 46, 46, 24, 9, 24, 20, 27, 61, 15, 15, 38, 44, 15, 46, 60, 56],
[30,34,29,33,0,29,14,14,28,27,56,37,35,13,63,49,60,48,69,69, 60, 32, 40, 39, 17, 10, 35, 7, 33, 41, 47, 7, 51, 16, 44, 45, 55, 8, 61, 30, 31, 26, 41, 24, 29, 28, 46, 48, 70, 67],
[55,5,61,64,29,0,21,46,53,37,61,17,13,5,15,40,38,30,25,5, 56, 39, 26, 33, 52, 65, 63, 13, 12, 63, 48, 7, 38, 64, 30, 24, 20, 18, 30, 44, 22, 46, 58, 11, 53, 47, 36, 66, 11, 44],
[48,44,24,66,14,21,0,58,47,34,21,58,59,61,35,69,60,10,20,58, 54, 54, 40, 23, 21, 23, 68, 20, 40, 18, 33, 14, 29, 57, 58, 64, 40, 58, 51, 7, 14, 20, 21, 16, 20, 23, 67, 27, 24, 31],
[30,44,13,38,14,46,58,0,31,35,67,68,55,42,65,37,15,26,24,41, 23, 45, 48, 18, 68, 48, 40, 62, 41, 6, 36, 58, 53, 42, 5, 52, 34, 48, 54, 58, 6, 56, 42, 10, 21, 13, 42, 40, 9, 70],
[24,47,49,46,28,53,47,31,0,9,9,16,67,58,51,69,70,70,14,68, 28, 52, 39, 22, 37, 61, 52, 18, 15, 29, 61, 50, 46, 14, 10, 62, 32, 37, 35, 39, 22, 20, 64, 64, 44, 29, 44, 49, 64, 10],
[27,27,18,52,27,37,34,35,9,0,40,9,25,64,60,5,47,58,21,9, 27, 45, 70, 47, 40, 20, 53, 63, 61, 44, 48, 65, 33, 24, 46, 24, 46, 8, 70, 18, 45, 27, 52, 6, 9, 66, 10, 26, 39, 21],
[65,37,62,6,56,61,21,67,9,40,0,22,11,36,32,30,63,32,23,8, 9, 26, 11, 16, 29, 7, 51, 47, 16, 9, 31, 35, 18, 62, 44, 6, 64, 34, 8, 12, 70, 62, 45, 6, 26, 35, 11, 42, 26, 34],
[35,51,54,42,37,17,58,68,16,9,22,0,19,20,6,51,5,45,31,21, 59, 34, 69, 43, 20, 58, 34, 21, 68, 14, 38, 69, 47, 45, 35, 43, 65, 29, 20, 32, 23, 34, 46, 40, 30, 62, 24, 31, 39, 9],
[34,63,32,8,35,13,59,55,67,25,11,19,0,48,26,11,40,15,63,48, 12, 37, 44, 10, 69, 48, 35, 62, 22, 59, 26, 67, 21, 68, 38, 42, 70, 55, 25, 8, 7, 58, 32, 30, 39, 63, 31, 44, 11, 27],
[34,8,5,32,13,5,61,42,58,64,36,20,48,0,60,55,58,46,51,50, 57, 18, 64, 54, 40, 69, 42, 19, 61, 61, 25, 26, 60, 37, 8, 56, 33, 23, 27, 29, 57, 7, 52, 62, 23, 48, 20, 46, 9, 68],
[67,11,14,35,63,15,35,65,51,60,32,6,26,60,0,26,40,62,8,38, 60, 38, 5, 65, 32, 53, 14, 58, 65, 56, 54, 31, 51, 42, 43, 27, 51, 15, 5, 42, 41, 41, 50, 66, 67, 64, 44, 66, 12, 27],
[19,47,23,6,49,40,69,37,69,5,30,51,11,55,26,0,33,64,46,69, 33, 11, 25, 56, 40, 55, 47, 63, 9, 16, 32, 53, 63, 6, 14, 32, 55, 39, 54, 16, 49, 45, 5, 53, 70, 13, 28, 17, 52, 16],
[63,55,62,9,60,38,60,15,70,47,63,5,40,58,40,33,0,43,60,35, 58, 61, 42, 54, 39, 43, 17, 38, 8, 8, 27, 46, 63, 49, 50, 50, 57, 48, 39, 26, 66, 21, 38, 53, 21, 60, 40, 65, 28, 66],
[6,70,44,65,48,30,10,26,70,58,32,45,15,46,62,64,43,0,40,16, 11, 29, 23, 22, 19, 5, 63, 34, 43, 25, 70, 69, 68, 48, 52, 27, 39, 23, 52, 56, 10, 19, 18, 46, 52, 60, 6, 18, 35, 29],
[27,13,26,40,69,25,20,24,14,21,23,31,63,51,8,46,60,40,0,46, 9, 55, 31, 49, 41, 8, 25, 68, 66, 29, 45, 42, 59, 68, 9, 63, 42, 17, 19, 51, 44, 43, 68, 43, 11, 15, 11, 14, 49, 47],
[55,29,23,29,69,5,58,41,68,9,8,21,48,50,38,69,35,16,46,0, 29, 15, 30, 57, 31, 54, 24, 47, 63, 18, 36, 12, 13, 9, 10, 46, 32, 47, 61, 55, 69, 56, 66, 53, 21, 52, 50, 56, 54, 26],
[31, 44, 11, 23, 60, 56, 54, 23, 28, 27, 9, 59, 12, 57, 60, 33, 58, 11, 9, 29, 0, 38, 56, 42, 67, 16, 64, 54, 23, 46, 35, 13, 41, 68, 34, 44, 12, 18, 15, 7, 22, 20, 28, 13, 70, 38, 30, 21, 29, 31],
[54, 35, 67, 51, 32, 39, 54, 45, 52, 45, 26, 34, 37, 18, 38, 11, 61, 29, 55, 15, 38, 0, 66, 37, 45, 5, 43, 56, 48, 39, 59, 68, 25, 57, 23, 5, 11, 26, 25, 6, 15, 34, 55, 49, 52, 35, 65, 70, 66, 35],
[37, 24, 5, 48, 40, 26, 40, 48, 39, 70, 11, 69, 44, 64, 5, 25, 42, 23, 31, 30, 56, 66, 0, 64, 30, 44, 41, 60, 23, 48, 44, 48, 25, 26, 17, 11, 23, 28, 40, 70, 6, 51, 64, 62, 12, 68, 34, 62, 62, 28],
[30, 13, 24, 32, 39, 33, 23, 18, 22, 47, 16, 43, 10, 54, 65, 56, 54, 22, 49, 57, 42, 37, 64, 0, 46, 62, 6, 10, 29, 14, 7, 56, 26, 58, 16, 49, 62, 54, 20, 28, 28, 22, 27, 10, 34, 35, 6, 47, 8, 16],
[9, 24, 8, 37, 17, 52, 21, 68, 37, 40, 29, 20, 69, 40, 32, 40, 39, 19, 41, 31, 67, 45, 30, 46, 0, 50, 41, 5, 24, 29, 14, 9, 35, 29, 59, 40, 58, 35, 5, 5, 15, 29, 52, 15, 23, 47, 23, 26, 15, 30],
[19, 15, 34, 67, 10, 65, 23, 48, 61, 20, 7, 58, 48, 69, 53, 55, 43, 5, 8, 54, 16, 5, 44, 62, 50, 0, 64, 63, 30, 49, 21, 67, 46, 18, 12, 42, 23, 9, 15, 8, 49, 56, 58, 22, 64, 45, 5, 27, 40, 46],
[40, 25, 57, 67, 35, 63, 68, 40, 52, 53, 51, 34, 35, 42, 14, 47, 17, 63, 25, 24, 64, 43, 41, 6, 41, 64, 0, 34, 58, 14, 69, 60, 27, 45, 25, 10, 39, 59, 55, 67, 27, 30, 14, 27, 41, 41, 15, 12, 31, 59],
[32, 12, 65, 53, 7, 13, 20, 62, 18, 63, 47, 21, 62, 19, 58, 63, 38, 34, 68, 47, 54, 56, 60, 10, 5, 63, 34, 0, 17, 38, 13, 62, 55, 42, 30, 37, 63, 64, 37, 39, 65, 43, 56, 48, 35, 19, 44, 17, 18, 61],
[64, 59, 40, 30, 33, 12, 40, 41, 15, 61, 16, 68, 22, 61, 65, 9, 8, 43, 66, 63, 23, 48, 23, 29, 24, 30, 58, 17, 0, 13, 47, 33, 45, 53, 25, 7, 60, 29, 48, 64, 23, 70, 60, 57, 48, 40, 30, 59, 44, 56],
[8, 69, 58, 7, 41, 63, 18, 6, 29, 44, 9, 14, 59, 61, 56, 16, 8, 25, 29, 18, 46, 39, 48, 14, 29, 49, 14, 38, 13, 0, 70, 57, 61, 37, 62, 27, 18, 37, 7, 59, 22, 49, 57, 50, 47, 20, 36, 66, 20, 18],
[34, 19, 42, 43, 47, 48, 33, 36, 61, 48, 31, 38, 26, 25, 54, 32, 27, 70, 45, 36, 35, 59, 44, 7, 14, 21, 69, 13, 47, 70, 0, 22, 56, 26, 44, 17, 10, 25, 20, 65, 66, 7, 44, 7, 33, 39, 60, 46, 21, 64],
[40, 51, 54, 42, 7, 7, 14, 58, 50, 65, 35, 69, 67, 26, 31, 53, 46, 69, 42, 12, 13, 68, 48, 56, 9, 67, 60, 62, 33, 57, 22, 0, 55, 42, 64, 29, 27, 45, 10, 29, 17, 58, 52, 49, 65, 59, 41, 6, 11, 38],
[13, 7, 5, 28, 51, 38, 29, 53, 46, 33, 18, 47, 21, 60, 51, 63, 63, 68, 59, 13, 41, 25, 25, 26, 35, 46, 27, 55, 45, 61, 56, 55, 0, 67, 60, 66, 69, 29, 26, 56, 19, 68, 66, 29, 27, 50, 52, 57, 44, 11],
[40, 45, 36, 63, 16, 64, 57, 42, 14, 24, 62, 45, 68, 37, 42, 6, 49, 48, 68, 9, 68, 57, 26, 58, 29, 18, 45, 42, 53, 37, 26, 42, 67, 0, 41, 7, 55, 48, 44, 19, 52, 52, 27, 7, 60, 55, 38, 51, 57, 26],
[61, 49, 21, 46, 44, 30, 58, 5, 10, 46, 44, 35, 38, 8, 43, 14, 50, 52, 9, 10, 34, 23, 17, 16, 59, 12, 25, 30, 25, 62, 44, 64, 60, 41, 0, 65, 43, 24, 22, 63, 29, 14, 67, 47, 69, 17, 59, 37, 43, 7],
[18, 7, 25, 46, 45, 24, 64, 52, 62, 24, 6, 43, 42, 56, 27, 32, 50, 27, 63, 46, 44, 5, 11, 49, 40, 42, 10, 37, 7, 27, 17, 29, 66, 7, 65, 0, 42, 46, 37, 8, 39, 8, 60, 44, 50, 47, 46, 67, 16, 36],
[15, 5, 68, 24, 55, 20, 40, 34, 32, 46, 64, 65, 70, 33, 51, 55, 57, 39, 42, 32, 12, 11, 23, 62, 58, 23, 39, 63, 60, 18, 10, 27, 69, 55, 43, 42, 0, 60, 37, 55, 16, 50, 65, 15, 33, 7, 19, 48, 62, 27],
[9, 14, 18, 9, 8, 18, 58, 48, 37, 8, 34, 29, 55, 23, 15, 39, 48, 23, 17, 47, 18, 26, 28, 54, 35, 9, 59, 64, 29, 37, 25, 45, 29, 48, 24, 46, 60, 0, 6, 64, 36, 65, 26, 50, 6, 38, 26, 27, 67, 54],
[48, 61, 39, 24, 61, 30, 51, 54, 35, 70, 8, 20, 25, 27, 5, 54, 39, 52, 19, 61, 15, 25, 40, 20, 5, 15, 55, 37, 48, 7, 20, 10, 26, 44, 22, 37, 37, 6, 0, 27, 46, 32, 43, 68, 45, 14, 60, 52, 12, 15],
[6, 23, 19, 20, 30, 44, 7, 58, 39, 18, 12, 32, 8, 29, 42, 16, 26, 56, 51, 55, 7, 6, 70, 28, 5, 8, 67, 39, 64, 59, 65, 29, 56, 19, 63, 8, 55, 64, 27, 0, 24, 32, 39, 23, 59, 57, 23, 45, 11, 49],
[24, 40, 42, 27, 31, 22, 14, 6, 22, 45, 70, 23, 7, 57, 41, 49, 66, 10, 44, 69, 22, 15, 6, 28, 15, 49, 27, 65, 23, 22, 66, 17, 19, 52, 29, 39, 16, 36, 46, 24, 0, 27, 49, 24, 49, 30, 30, 18, 52, 59],
[69, 39, 70, 61, 26, 46, 20, 56, 20, 27, 62, 34, 58, 7, 41, 45, 21, 19, 43, 56, 20, 34, 51, 22, 29, 56, 30, 43, 70, 49, 7, 58, 68, 52, 14, 8, 50, 65, 32, 32, 27, 0, 66, 38, 17, 44, 18, 32, 61, 33],
[55, 42, 13, 15, 41, 58, 21, 42, 64, 52, 45, 46, 32, 52, 50, 5, 38, 18, 68, 66, 28, 55, 64, 27, 52, 58, 14, 56, 60, 57, 44, 52, 66, 27, 67, 60, 65, 26, 43, 39, 49, 66, 0, 66, 29, 51, 43, 11, 32, 66],
[21, 26, 68, 15, 24, 11, 16, 10, 64, 6, 6, 40, 30, 62, 66, 53, 53, 46, 43, 53, 13, 49, 62, 10, 15, 22, 27, 48, 57, 50, 7, 49, 29, 7, 47, 44, 15, 50, 68, 23, 24, 38, 66, 0, 8, 30, 20, 20, 28, 65],
[50, 32, 13, 38, 29, 53, 20, 21, 44, 9, 26, 30, 39, 23, 67, 70, 21, 52, 11, 21, 70, 52, 12, 34, 23, 64, 41, 35, 48, 47, 33, 65, 27, 60, 69, 50, 33, 6, 45, 59, 49, 17, 29, 8, 0, 38, 58, 28, 35, 67],
[14, 32, 17, 44, 28, 47, 23, 13, 29, 66, 35, 62, 63, 48, 64, 13, 60, 60, 15, 52, 38, 35, 68, 35, 47, 45, 41, 19, 40, 20, 39, 59, 50, 55, 17, 47, 7, 38, 14, 57, 30, 44, 51, 30, 38, 0, 13, 9, 51, 66],
[46, 69, 47, 15, 46, 36, 67, 42, 44, 10, 11, 24, 31, 20, 44, 28, 40, 6, 11, 50, 30, 65, 34, 6, 23, 5, 15, 44, 30, 36, 60, 41, 52, 38, 59, 46, 19, 26, 60, 23, 30, 18, 43, 20, 58, 13, 0, 12, 48, 14],
[61, 8, 13, 46, 48, 66, 27, 40, 49, 26, 42, 31, 44, 46, 66, 17, 65, 18, 14, 56, 21, 70, 62, 47, 26, 27, 12, 17, 59, 66, 46, 6, 57, 51, 37, 67, 48, 27, 52, 45, 18, 32, 11, 20, 28, 9, 12, 0, 50, 46],
[64, 31, 10, 60, 70, 11, 24, 9, 64, 39, 26, 39, 11, 9, 12, 52, 28, 35, 49, 54, 29, 66, 62, 8, 15, 40, 31, 18, 44, 20, 21, 11, 44, 57, 43, 16, 62, 67, 12, 11, 52, 61, 32, 28, 35, 51, 48, 50, 0, 7],
[27, 57, 48, 56, 67, 44, 31, 70, 10, 21, 34, 9, 27, 68, 27, 16, 66, 29, 47, 26, 31, 35, 28, 16, 30, 46, 59, 61, 56, 18, 64, 38, 11, 26, 7, 36, 27, 54, 15, 49, 59, 33, 66, 65, 67, 66, 14, 46, 7, 0]
])

n_c = len(dst)
city_demands = np.array([0, 20, 20, 20, 20])
city_demands = np.array([0, 10, 10, 10, 10,10,10,10,10,10])
city_demands = np.array([0,9,9,9,9,9,9,9,9,9,9,9,9,9,9])
city_demands = np.zeros((1,n_c))  # for 20 city
#city_demands = np.array([0, 6, 3, 1, 1, 7, 9, 1, 9, 3, 3, 2, 7, 1, 6, 1, 8, 9, 5, 3])
n_agents = 3
v_capacity_default = [42] * n_agents
v_capacity_default = [ 29, 28, 28]
init_city_default = [0] * n_agents
v_capacity_1=[]
v_capacity_new_1=[]
init_city_1=[]
current_city_1 = []
current_city_one_hot = [0] * n_c
total_reward_1 =[]
full_filled_cities_1=[]
total_reward = 0
demands_random = True

for vheical in range(n_agents):
    v_capacity_1.append(v_capacity_default[vheical])
    v_capacity_new_1.append(v_capacity_1[vheical])
    init_city_1.append(init_city_default[vheical])
    current_city_1.append(init_city_1[vheical])
    total_reward_1.append(0)
    full_filled_cities_1.append([])

city_demands_new = city_demands.copy() 
env = environment(dst, city_demands_new)

agents=[]
for k in range(n_agents):
    agents+=[agent(v_capacity_1[k], full_filled_cities_1[k], init_city_1[k], current_city_1[k], total_reward)]

city_demands_new, v_capacity_1, init_city_1 = env.reset(city_demands,init_city = 0, city_demands_random=demands_random, init_city_random=False)

for k in range(n_agents):
    v_capacity_new_1[k], current_city_1[k], total_reward_1[k], full_filled_cities_1[k] = agents[k].reset(v_capacity_1[k], init_city_1[k], init_city_1[k])
#v_capacity_new_2, current_city_2, total_reward_2, full_filled_cities_2 = agents[1].reset(v_capacity_2, init_city_2, current_city_2)

DQN_models = []
n_models = 50
discard_period = 1000  # episode
save_models_period = 1000
n_best_model = 25
n_random_model = n_models - n_best_model

input_layer = Input(shape=2*n_c+2)
x = Dense(500, activation='linear')(input_layer)
x = Dense(500, activation='linear')(x)
x = Dense(500, activation='linear')(x)
x = Dense(500, activation='linear')(x)
x = Dense(500, activation='linear')(x)
x = Dense(500, activation='linear')(x)
output_layer = Dense(n_c, activation='tanh')(x)

model1 = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model_target = tf.keras.Model(inputs=input_layer, outputs=output_layer)
#model1 = keras_models.load_model('VRP_DQN_model1_MA_20city_3agent_02.h5')
m1_weights = model1.get_weights()
model_target.set_weights(m1_weights)
#model_static = keras_models.load_model('vhr_10city_10M.h5')
#opt = Adam(learning_rate=1e-4)
model1.compile(loss='mean_squared_error', optimizer='sgd', metrics='mean_squared_error')
#model_target = model1

for kkk in range(n_models):
    input_layer = Input(shape=2 * n_c + 2)
    x = Dense(500, activation='linear')(input_layer)
    x = Dense(500, activation='linear')(x)
    x = Dense(500, activation='linear')(x)
    x = Dense(500, activation='linear')(x)
    x = Dense(500, activation='linear')(x)
    x = Dense(500, activation='linear')(x)
    output_layer = Dense(n_c, activation='tanh')(x)
    model_X = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    DQN_models += [{'model': model_X, 'best_path': -5000, 'path_loss': -5000, 'n_trials': 0, 'n_wins': 0, 'routs': []}]

total_reward_list = []
windowed_reward_list = []
dqn_performance_wins_list = []
dqn_performance_trials_list = []

'''
with open('total_reward_list_20city_3gent_02.txt', 'r') as fp:
    for line in fp:
        line = line.strip()
        line = float(line)
        total_reward_list.append(line)

with open('windowed_reward_list_20city_3gent_02.txt', 'r') as fp:
    for line in fp:
        line = line.strip()
        line = float(line)
        windowed_reward_list.append(line)
'''

#nn_input_state_1 = np.concatenate((city_demands_new, np.array([v_capacity_new_1[k]]),np.array([init_city_1[k]]),np.array([current_city_1[k]])))
current_city_one_hot[current_city_1[k]] = 1
nn_input_state_1 = np.concatenate((city_demands_new, np.array([v_capacity_new_1[k]]),np.array([init_city_1[k]]),np.array(current_city_one_hot)))
buffer_input1 = nn_input_state_1.reshape(1,-1)
buffer_q_s1 = model1(nn_input_state_1.reshape(1,-1))
buffer_input = np.array(buffer_input1)
buffer_q_s1 = model1(nn_input_state_1.reshape(1,-1))
buffer_q_s = buffer_q_s1
buffer_input_LT = buffer_input
buffer_q_s_LT = buffer_q_s

for k in range(1, n_agents):
    current_city_one_hot = [0] * n_c
    current_city_one_hot[current_city_1[k]] = 1
    nn_input_state_1 = np.concatenate((city_demands_new, np.array([v_capacity_new_1[k]]), np.array([init_city_1[k]]),
                                       np.array(current_city_one_hot)))
    #nn_input_state_1 = np.concatenate((city_demands_new, np.array([v_capacity_new_1[k]]),np.array([init_city_1[k]]),np.array([current_city_1[k]])))
    buffer_input1 = nn_input_state_1.reshape(1,-1)
    buffer_q_s1 = model1(nn_input_state_1.reshape(1,-1))
    buffer_input = np.vstack((buffer_input, buffer_input1.tolist()))
    buffer_q_s = np.vstack((buffer_q_s, buffer_q_s1[0]))

buffer_size = n_c + 3*n_agents
train_period = n_c + 3*n_agents
buffer_size_LT = 10000
LT_buffering = False
test_period = 1
target_model_update_period = 1000
n_update =1

alpha = 0.5  # learning rate
gamma = 0.99  # discount factor
epsilon = 0.7  #
force_elite_actions = 0.0
elite_running_flag = False
elite_index = 0
grand_total_reward = 0
windowed_reward = 0
window_size = 10

best_total_reward = -5000
best_actions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0]
n_test = 0
rew = 0

for episode in range(1, 100000000):

    step = -1
    done = False
    pre_rep_index = 0

    bf_state = []
    bf_q_s = []
    bf_q_s_next = []
    bf_action = []
    bf_rew = []

    while not done:
        for agent in agents:
            if not agent.done and not done:
                step += 1
                v_capacity_new, init_city, current_city = agent.info()
                nn_input_state = np.concatenate((city_demands_new, np.array([agent.v_capacity_new]),np.array([agent.init_city]),np.array([agent.current_city])))
                current_city_one_hot = [0] * n_c
                current_city_one_hot[agent.current_city] = 1
                city_demands_new_3_state = list(map(float, (agent.v_capacity_new >= city_demands_new)))
                for k in range(len(city_demands_new)):
                    if city_demands_new[k] == 0:
                        city_demands_new_3_state[k] = -1
                v_cap = int(agent.v_capacity_new > 0)
                nn_input_state_one_hot = np.concatenate((city_demands_new_3_state, [v_cap],
                                                 np.array([agent.init_city]), np.array(current_city_one_hot)))

                action_vals_1 = model1(nn_input_state_one_hot.reshape(1,-1))
                action_vals_2 = list(np.zeros((1,n_c)))
                index_zeros = np.where(city_demands_new == 0)
                if agent.v_capacity_new == 0:
                    action_vals_2[0][0] = 1000
                elif index_zeros[0].size > 1:
                    for indexes in index_zeros[0]:
                        action_vals_2[0][indexes] = -1000

                # Searching the best model
                action_vals_list = np.zeros((n_models, n_c))
                prob_act_list = np.zeros((n_models, n_c))
                prob_max = np.zeros((1, n_models))
                prob_act = np.zeros((1, n_models))
                for k in range(n_models):
                    model = DQN_models[k]['model']
                    action_vals_swarm = model(nn_input_state_one_hot.reshape(1, -1))
                    action_vals_list[k, :] = action_vals_swarm + action_vals_2
                    prob_act_list[k, :] = softmax(action_vals_list[k,:])
                    prob_max[0, k] = np.max(prob_act_list[k, :])

                model_to_use_index = np.argmax(prob_max)
                prob_act[0,:] = prob_act_list[model_to_use_index, :]

                pp = np.random.rand(1)
                sum_prop = 0
                for k in range(n_c):
                    sum_prop += prob_act[0,k]
                    if pp <= sum_prop:
                        action = k
                        break

                epsilon_greedy_force_elite_prob = np.random.rand(1)
                if not elite_running_flag:
                    if epsilon_greedy_force_elite_prob < epsilon:
                        #action = env.action_space.sample()
                        action = action
                    elif epsilon_greedy_force_elite_prob > 1-force_elite_actions:
                        elite_running_flag = True
                        print('elite action training started')
                        city_demands_new, v_capacity_1, init_city_1 = env.reset(city_demands,init_city = 0, city_demands_random=False, init_city_random=False)
                        for k in range(n_agents):
                            v_capacity_new_1[k], current_city_1[k], total_reward_1[k], full_filled_cities_1[k] = agents[k].reset(v_capacity_1[k], init_city_1[k], init_city_1[k])

                        step = 0
                        elite_index = 0
                    else:
                        action = action

                if elite_running_flag:
                    action = best_actions[elite_index]
                    elite_index += 1
        
                if elite_index >= len(best_actions):
                    elite_running_flag = False

                nn_input = nn_input_state_one_hot

                rew, done, city_demands_new, cities = env.action_space.step(action,agent,agents)
                
                action_vals_model_target = model_target(nn_input_state_one_hot.reshape(1,-1))
                
                bf_state.append(nn_input_state_one_hot)
                bf_q_s.append(action_vals_1)
                bf_q_s_next.append(action_vals_model_target)
                bf_action.append(action)
                sig = 1-1/(1+np.exp(1e-2*rew))
                rew_ts = 2*sig - 1
                bf_rew.append(rew_ts)

    istep = -1
    for k in range(step+1):
        nn_input = bf_state[k]
        q_s = np.array(bf_q_s[k])
        action = bf_action[k]
        if k == step:
            q_next = 0
        else:
            q_next = bf_q_s_next[k+1]
        
        rew = bf_rew[k]
        error = rew + gamma * np.max(q_next) - q_s[0][action]
        q_s[0][action] =  q_s[0][action] + alpha * error # Update Q-value using Bellman equation
        # Buffering
        if len(nn_input) != 0:
            if buffer_input.shape[0] >= buffer_size:
                rep_index = k % buffer_size
                buffer_input[rep_index] = nn_input
                buffer_q_s[rep_index] = q_s[0]
            else:
                #buffer_input = np.vstack((buffer_input, nn_input), axis=0)
                buffer_input = np.vstack((buffer_input, nn_input.tolist()))
                buffer_q_s = np.vstack((buffer_q_s, q_s[0]))
                rep_index = buffer_q_s.shape[0]-1

    model1.fit(buffer_input[pre_rep_index:rep_index+1][:], buffer_q_s[pre_rep_index:rep_index+1][:], epochs=1, batch_size=1, verbose=1)
    pre_rep_index = rep_index + 1
    if LT_buffering:
        if len(buffer_input_LT) <= 1:
            buffer_input_LT = buffer_input[:rep_index+1][:]
            buffer_q_s_LT = buffer_q_s[:rep_index+1][:]
        else:
            buffer_input_LT = np.vstack((buffer_input_LT,buffer_input[:rep_index+1][:]))
            buffer_q_s_LT = np.vstack((buffer_q_s_LT,buffer_q_s[:rep_index+1][:]))

    if LT_buffering:
        if len(buffer_input_LT) > buffer_size_LT:
            model1.fit(buffer_input_LT, buffer_q_s_LT, epochs=1, batch_size=1, verbose=1)
            #print('Long term training done')
            buffer_input_LT = np.array([])
            buffer_q_s_LT = np.array([])

    if episode % target_model_update_period == 0:
        n_update += 1
        #target_model_update_period = 20 * n_update**2
        m1_weights = model1.get_weights()
        model_target.set_weights(m1_weights)
        #model_target = model1
        print('Episode {}'.format(episode))
        print('Target model updated')

    if episode % save_models_period == 0:
        for n_saves in range(n_models):
            if n_saves < 10:
                model_number = '0' + str(n_saves)
            else:
                model_number = str(n_saves)
            model_name = 'xx_crvp_DQN_model_' + model_number + '.h5'
            model_to_save = DQN_models[n_saves]['model']
            model_to_save.save(model_name)

        model1.save('xx_crvp_DQN_model1.h5')
        #model_target.save('VRP_DQN_model_target_MA_5city_2agent_02.h5')

    total_reward = 0
    for agent in agents:
        total_reward += agent.total_reward

    if total_reward > best_total_reward:
        best_total_reward = total_reward
        best_routs=[]
        for agent in agents:    
            best_routs.append(agent.cities)
        print('**********___________________BEST_RESULT__________________________************')
        print('city route: {},  Reward: {} \n routs: {}'.format(cities, best_total_reward,best_routs))
        print('******************************************************************************')
        best_actions = cities

    #print('modes saved')
    city_demands_new, v_capacity_1, init_city_1 = env.reset( city_demands,init_city = init_city_1[0], city_demands_random=demands_random, init_city_random=False)
    city_demands_dqn = city_demands_new.copy()
    for k in range(n_agents):
        v_capacity_new_1[k], current_city_1[k], total_reward_1[k], full_filled_cities_1[k] = agents[k].reset(v_capacity_1[k], init_city_1[k], init_city_1[k])

    if episode % test_period == 0:
        n_test += 1
        max_step = 100
        step = 0
        done = False
        print('******************____TESTING_____*************************')
        print('city_demands_new: {},  v_capacity_1: {}'.format(city_demands_new, v_capacity_1))

        while not done:
            for agent in agents:
                if not agent.done and not done:
                    step += 1


                    if np.random.rand(1) < 0:
                        action = env.action_space.sample()
                    else:
                        nn_input_state = np.concatenate((city_demands_new, np.array([agent.v_capacity_new]),np.array([agent.init_city]),np.array([agent.current_city])))
                        current_city_one_hot = [0] * n_c
                        current_city_one_hot[agent.current_city] = 1
                        city_demands_new_3_state = list(map(float, (agent.v_capacity_new >= city_demands_new)))
                        for k in range(len(city_demands_new)):
                            if city_demands_new[k] == 0:
                                city_demands_new_3_state[k] = -1
                        v_cap = int(agent.v_capacity_new > 0)
                        nn_input_state_one_hot = np.concatenate((city_demands_new_3_state, [v_cap],
                                                                 np.array([agent.init_city]),
                                                                 np.array(current_city_one_hot)))
                        action_vals_1 = model1(nn_input_state_one_hot.reshape(1,-1))
                        action_vals_2 = list(np.zeros((1, n_c)))
                        index_zeros = np.where(city_demands_new == 0)
                        if agent.v_capacity_new == 0:
                            action_vals_2[0][0] = 1000
                        elif index_zeros[0].size > 1:
                            for indexes in index_zeros[0]:
                                action_vals_2[0][indexes] = -1000

                        action_vals = action_vals_1 + action_vals_2
                        action = np.argmax(action_vals)

                    rew, done, city_demands_new, cities = env.action_space.step(action,agent,agents)
                    #current_city = next_city
        total_reward_test = 0

        routs=[]
        for agent in agents:
            total_reward_test += agent.total_reward
            routs.append(agent.cities)

        for number, my_dqn_dic in enumerate(DQN_models):
            city_demands_new, v_capacity_1, init_city_1 = env.reset(city_demands_dqn, init_city=init_city_1[0],
                                                                    city_demands_random=False,
                                                                    init_city_random=False)
            for k in range(n_agents):
                v_capacity_new_1[k], current_city_1[k], total_reward_1[k], full_filled_cities_1[k] = agents[k].reset(
                    v_capacity_1[k], init_city_1[k], init_city_1[k])

            my_model = my_dqn_dic['model']
            # my_path_loss = my_dqn_dic['path_loss']
            my_dqn_dic['n_trials'] += 1
            my_n_wins = my_dqn_dic['n_wins']

            step = 0
            done = False

            while not done:
                for agent in agents:
                    if not agent.done and not done:
                        step += 1

                        if np.random.rand(1) < 0:
                            action = env.action_space.sample()
                        else:
                            nn_input_state = np.concatenate((city_demands_new, np.array([agent.v_capacity_new]),
                                                             np.array([agent.init_city]),
                                                             np.array([agent.current_city])))
                            current_city_one_hot = [0] * n_c
                            current_city_one_hot[agent.current_city] = 1
                            city_demands_new_3_state = list(map(float, (agent.v_capacity_new >= city_demands_new)))
                            for k in range(len(city_demands_new)):
                                if city_demands_new[k] == 0:
                                    city_demands_new_3_state[k] = -1
                            v_cap = int(agent.v_capacity_new > 0)
                            nn_input_state_one_hot = np.concatenate((city_demands_new_3_state, [v_cap],
                                                                     np.array([agent.init_city]),
                                                                     np.array(current_city_one_hot)))
                            action_vals_1 = my_model(nn_input_state_one_hot.reshape(1, -1))
                            action_vals_2 = list(np.zeros((1, n_c)))
                            index_zeros = np.where(city_demands_new == 0)
                            if agent.v_capacity_new == 0:
                                action_vals_2[0][0] = 1000
                            elif index_zeros[0].size > 1:
                                for indexes in index_zeros[0]:
                                    action_vals_2[0][indexes] = -1000

                            action_vals = action_vals_1 + action_vals_2
                            action = np.argmax(action_vals)

                        rew, done, city_demands_new, cities = env.action_space.step(action, agent, agents)
                        # current_city = next_city
            total_reward_dqns = 0

            routs = []
            for agent in agents:
                total_reward_dqns += agent.total_reward
                routs.append(agent.cities)

            DQN_models[number]['path_loss'] = total_reward_dqns
            DQN_models[number]['routs'] = routs

        # Evaluation
        evaluation_list = []
        for nn_models in range(n_models):
            evaluation_list.append(DQN_models[nn_models]['path_loss'])
        best_model_index = np.argmax(np.array(evaluation_list))
        DQN_models[best_model_index]['n_wins'] += 1
        print('episode: {}, Reward: {} \n routs: {}'.format(episode, DQN_models[best_model_index]['path_loss'], DQN_models[best_model_index]['routs']))
        print('***********************************************************')

        total_reward_list.append(DQN_models[best_model_index]['path_loss'])
        if len(total_reward_list) >= window_size:
            windowed_reward = np.sum(total_reward_list[-window_size:]) / window_size
            windowed_reward_list.append(windowed_reward)
            # visualize_loss(windowed_reward_list)
            with open(r'xx__list_50city_3gent_XX.txt', 'w') as fp:
                fp.write('\n'.join(str(item) for item in total_reward_list))
            with open(r'xx_windowed_reward_list_50city_3gent_XX.txt', 'w') as fp:
                fp.write('\n'.join(str(item) for item in windowed_reward_list))

        # Discard Best
        for nn_best_models in range(n_best_model):
            if total_reward_test > DQN_models[nn_best_models]['best_path']:
                #model_best = tf.keras.Model(inputs=input_layer, outputs=output_layer)
                model_best = DQN_models[nn_best_models]['model']
                m1_weights = model1.get_weights()
                model_best.set_weights(m1_weights)
                DQN_models[nn_best_models]['best_path'] = total_reward_test
                DQN_models[nn_best_models]['path_loss'] = total_reward_test
                DQN_models[nn_best_models]['n_trials'] = 0
                DQN_models[nn_best_models]['n_wins'] = 0
                DQN_models[nn_best_models]['routs'] = []
                break

        # Discard worst model in randomly selected models
        if episode % discard_period == 0:
            dqn_performance = []
            for nn_random_models in range(n_best_model, n_models):
                dqn_performance.append(DQN_models[nn_random_models]['n_wins'] / DQN_models[nn_random_models]['n_trials'])
            worst_model_index = np.argmin(np.array(dqn_performance))
            model_random = DQN_models[n_best_model+worst_model_index]['model']
            m1_weights = model1.get_weights()
            model_random.set_weights(m1_weights)
            DQN_models[n_best_model+worst_model_index]['best_path'] = -1000
            DQN_models[n_best_model+worst_model_index]['path_loss'] = total_reward_test
            DQN_models[n_best_model+worst_model_index]['n_trials'] = 0
            DQN_models[n_best_model+worst_model_index]['n_wins'] = 0
            DQN_models[n_best_model+worst_model_index]['routs'] = []

        dqn_performance = []
        dqn_performance_wins = []
        dqn_performance_trials = []
        for nn_random_models in range(n_models):
            dqn_performance_wins.append(DQN_models[nn_random_models]['n_wins'])
            dqn_performance_trials.append(DQN_models[nn_random_models]['n_trials'])

        if len(total_reward_list) >= window_size:
            dqn_performance_wins_list.append(dqn_performance_wins)
            dqn_performance_trials_list.append(dqn_performance_trials)
            # visualize_loss(windowed_reward_list)
            with open(r'xx__wins_50agent_50city_3vehicle.txt', 'w') as fp:
                fp.write('\n'.join(str(item) for item in dqn_performance_wins_list))
            with open(r'xx_trials_50agent_50city_3vehicle.txt', 'w') as fp:
                fp.write('\n'.join(str(item) for item in dqn_performance_trials_list))

    city_demands_new, v_capacity_1, init_city_1 = env.reset(city_demands,init_city = init_city_1[0], city_demands_random=demands_random, init_city_random=False)

    for k in range(n_agents):
        v_capacity_new_1[k], current_city_1[k], total_reward_1[k], full_filled_cities_1[k] = agents[k].reset(v_capacity_1[k], init_city_1[k], init_city_1[k])


    