import torch
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from dqn_cartpole import QNetwork, select_action

num_eval_episodes = 4

# Testing the trained agent
env = gym.make("CartPole-v1", render_mode = "rgb_array")
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="eval",
                  episode_trigger=lambda x: True)

observation_size = env.observation_space.shape[0] # type: ignore
action_size = env.action_space.n  # type: ignore
q_net = QNetwork(observation_size, action_size)
q_net.load_state_dict(torch.load(r'C:\Users\srish\RL\OpenAI_gym\model\2025-02-04_23-07-28\cartpole-dqn.pth', weights_only=True))
q_net.eval()

for episode_num in range(num_eval_episodes):
    state, _ = env.reset()
    done = False
    #total_reward = 0.0
    while not done:
        action = select_action(epsilon=0.0, env=env, state=state, q_net=q_net)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        #total_reward += float(reward)

env.close()
