import torch
import random 
from torch import nn
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as func

class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128) 
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = func.relu(self.layer1(x))
        x = func.relu(self.layer2(x))
        return self.layer3(x) # output size = [n_observations, n_actions]

class Epsilon():
    def __init__(self, start, end, training_steps):
        self.start = start
        self.end = end
        self.epsilon = start
        self.training_steps = training_steps

    def calculate_decay(self):
        decay_factor = (self.end/self.start)**(1/self.training_steps)
        return decay_factor
    
    def calculate_epsilon(self):
        decay_factor = self.calculate_decay()
        self.epsilon = self.epsilon * decay_factor
        return self.epsilon


def select_action(state, Qout):
    n = np.random.rand()

    if n < epsilon:
        action = env.action_space.sample()

    else:
        action = torch.argmax(Qvalue)

    # Qvalue = torch.sum(Qout, 1)
    # epsilon = ep_start
    # action = random.choice([0,1])
    # action = torch.argmax(Qvalue)

    return action
    

def main(no_eps: int):

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    buffer = []
    n_actions = env.action_space.n  # type: ignore
    state = env.reset(seed=11)
    state = state[0]
    n_observations = state.size
    ep_start = 1.0 #in the beginning, epsilon=1
    ep_end=0.001 
    decay_steps = 100
    epsilon = ep_start
    # epsilon = Epsilon(ep_start, ep_end, 100)
    lr = 1e-4
    policy = QNetwork(n_observations, n_actions)
    target = QNetwork(n_observations, n_actions)
    # initialize weights of the policy and the target network
    target.load_state_dict(policy.state_dict())
    # set optimizer
    optimizer = optim.Adam(policy.parameters(), lr = lr)
    training_step = 0 
    for ep in tqdm(range(no_eps), desc = "Episodes completed"):
        term, trunc = False, False
        state = env.reset(seed=11)
        state = state[0]
        while term == False and trunc == False:
            action = select_action(state)

            action = env.action_space.sample()
            new_s, reward, term, trunc, _ = env.step(action)
            buffer.append((state, action, reward, new_s))
            state = new_s
            step +=1
        ep +=1


        training_step +=1



if __name__ == "__main__":

    main(no_eps = 1000)



