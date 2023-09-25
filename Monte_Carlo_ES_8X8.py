import gym
import numpy as np
from tqdm import tqdm
import argparse
import imageio 

def main(gamma, n_eps):

    # create environment
    env = gym.make("FrozenLake8x8-v1", desc=None, is_slippery=False)

    n_states = env.observation_space.n #no. of states
    n_actions = env.action_space.n #no. of actions
    actions = env.action_space #object
    qtable = np.zeros((n_states, n_actions))
    returns = {}
    for n in range(n_states):
        for m in range(n_actions):
            returns.update({(n, m): []})
    gamma = gamma
    n_eps = n_eps
    #print(env.desc) #environment map
    
    for p in tqdm(range(n_eps), desc = "Episodes completed"):
        episode = [] # a list that contains all the state action pairs of an episode
        rewards = [] # a list that contains all the rewards of an episode
        state = env.reset() #resets the environment to its initial state
        #state = env.observation_space.sample() #choose a random state
        done = False
        while not done: # creating an episode 
            action = actions.sample() #choosing a random action
            new_s, reward, done, prob = env.step(action) #performing a step in the env using the randomly chosen action
            episode.append((state, action)) 
            rewards.append(reward)
            state = new_s
            #env.render(mode='human') 
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
            
            policy_array = np.argmax(qtable, axis = 1)

    frames = []
    state = env.reset()
    policy = []
    rewards = []
    done = False
    while not done:
        action = policy_array[state]
        new_s, reward, done, prob = env.step(action)
        policy.append((state, action))
        rewards.append(reward)
        state = new_s
        frame = env.render(mode= 'rgb_array')
        frames.append(frame)
        #env.render(mode='human') # to render environment
    env.close()
    imageio.mimsave('FrozenLake8x8_2.gif', frames, fps = 5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Runs Monte Carlo Exploring Starts algorithm \
                                     Frozen lake environemnt')
    parser.add_argument('gamma', type = float, default=0.9, help = 'discount factor of rewards')
    parser.add_argument('n_eps', type = int, default = 2000, help = 'No. episodes generated for learning')
    args = parser.parse_args()
    main(**vars(args))