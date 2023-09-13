import gym
import numpy as np
import time
import random

if __name__ == '__main__':

    # create environment
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

    n_states = env.observation_space.n #no. of states
    n_actions = env.action_space.n #no. of actions
    actions = env.action_space #object
    qtable = np.zeros((n_states, n_actions))
    returns = {}
    for n in range(n_states):
        for m in range(n_actions):
            returns.update({(n, m): []})
    gamma = 0.7
    n_eps = 10
    print(env.desc) #environment map
    for p in range(n_eps):
        episode = [] # a list that contains all the state action pairs of an episode
        rewards = [] # a list that contains all the rewards of an episode
        state = env.reset() #choose a random state
        #state = env.observation_space.sample() #choose a random state
        done = False
        while not done: # creating an episode 
            action = actions.sample() #choosing a random action
            new_s, reward, done, prob = env.step(action) #performing a step in the env using the randomly chosen action
            episode.append((state, action)) 
            rewards.append(reward)
            state = new_s
        G = 0
        for q in range(-1, -len(episode)-1, -1): #reversed returns an iterable object, doesn't reverse the original list
            G = gamma*G + rewards[q]
            indices = [r for r in range(-1, -len(episode)-1, -1) if episode[r] == episode[q]]
            if len(indices) == 1 or q == min(indices):
                returns[episode[q]] += [G]
                
        for key, value in returns.items():
            if len(value)!= 0:
                s, t = key
                qtable[s, t] = np.average(value)

        