# %%
import torch

# %%
import os, datetime
# %%
import random 
from torch import nn
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.tensorboard.writer import SummaryWriter
from matplotlib import pyplot as plt
from matplotlib import animation

class QNetwork(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.layer1 = nn.Linear(observation_size, 128)
        self.layer2 = nn.Linear(128, 128) 
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x) # output size = [observation_size, action_size]

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


def select_action(epsilon, env, state, q_net):
    # Epsilon-greedy action selection
    n = np.random.rand()
    if n <= epsilon:
        action = env.action_space.sample()
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state_tensor)
        action = q_values.argmax().item()    

    return action

def save_model(q_net):
    model_dir = os.path.join('model', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(model_dir, exist_ok=True)
    model_path=os.path.join(model_dir, 'cartpole-dqn.pth')
    torch.save(q_net.state_dict(), model_path)

def main():

    BUFFER_SIZE = 10000
    LR = 1e-4
    BATCH_SIZE = 64
    EPSILON_START = 0.9
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.99
    GAMMA = 0.99  # discount factor
    TAU = 0.005    # Soft update parameter
    EPISODES = 500 

    # TODO: Record video
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # initialize replay buffer
    buffer = ReplayBuffer(BUFFER_SIZE)
    # TODO: Check if Summary writer is working correctly
    log_dir = os.path.join(os.getcwd(), 'runs', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    writer = SummaryWriter(log_dir=log_dir)

    observation_size = env.observation_space.shape[0]  # type: ignore
    action_size = env.action_space.n  # type: ignore

    q_net = QNetwork(observation_size, action_size)
    target_net = QNetwork(observation_size, action_size)
    # initialize weights of the q_net and the target_net network
    target_net.load_state_dict(q_net.state_dict())
    # set optimizer
    optimizer = optim.Adam(q_net.parameters(), lr = LR)
    epsilon = EPSILON_START

    for ep in tqdm(range(EPISODES), desc = "Episodes completed"):
        done = False
        state, _ = env.reset()
        total_reward = 0 

        while not done:

            action = select_action(epsilon, env, state, q_net)

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, new_state, done)
            state = new_state
            total_reward += float(reward)

            # Train when buffer has enough samples
            if len(buffer) >= BATCH_SIZE:
              states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)  
              states = torch.FloatTensor(states)
              actions = torch.LongTensor(actions)
              rewards = torch.FloatTensor(rewards)
              next_states = torch.FloatTensor(next_states)
              dones = torch.FloatTensor(dones)

              # Compute Q-values and target
              current_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
              next_q = target_net(next_states).max(1)[0].detach()
              target_q = rewards + GAMMA * next_q * (1 - dones)

              # Optimize the model
              loss = nn.MSELoss()(current_q, target_q)
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

              # Log loss to TensorBoard
              writer.add_scalar("Loss", loss.item(), ep)

              # Soft update target network
              for target_param, q_param in zip(target_net.parameters(), q_net.parameters()):
                target_param.data.copy_(TAU * q_param.data + (1 - TAU) * target_param.data)

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Log episode reward and epsilon to TensorBoard
        writer.add_scalar("Episode Reward", total_reward, ep)
        writer.add_scalar("Epsilon", epsilon, ep)
        
        # Monitor progress
        if (ep + 1) % 10 == 0:
            print(f"Episode: {ep+1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    print("Saving trained model")
    save_model(q_net)
    writer.close()
    env.close()

if __name__ == "__main__":

    main()