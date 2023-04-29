from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class VanillaNN(nn.Module):
    
    def __init__(self, n_observations, n_actions):
        super(VanillaNN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=-1)  # dim need to be -1 to prevent NaN results
        return x


# TODO: support GPU training by passing data to "cuda" device when inference
class Reinforce:
    def __init__(self, env, gamma=0.99, learning_rate=3e-3, device="cpu"):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.device = device

        # get number of actions and observations
        self._n_observations = self.env.observation_space.shape[0]
        self._n_actions = self.env.action_space.n

        # setup NN model
        self.policy_net = VanillaNN(self._n_observations, self._n_actions).to(self.device)

        # setup optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = self.loss_function
        
    def loss_function(self, prob_batch, expected_returns_batch):
        return - torch.sum(torch.log(prob_batch) * expected_returns_batch)

    def get_trajectory(self, max_steps=500, render=False):
        trajectory = list()
        done = False  # incase of early loop-termination (max_steps) before the environment terminated
        state = self.env.reset()

        for _ in range(max_steps):
            action_probs = self.policy_net(torch.from_numpy(state).float())
            # TODO: support continuous action space
            selected_action = np.random.choice(np.array(range(self._n_actions)), p=action_probs.data.numpy())
            next_state, reward, done, _ = self.env.step(selected_action)
            trajectory.append((state, selected_action, reward))
            prev_state = state
            state = next_state
            
            if render:
                self.env.render()

            if done:
                break

        return trajectory, done

    def train(self, num_episodes=1000, max_steps=500):
        for episode in tqdm(range(num_episodes)):
            # get sample trajectory from the policy network
            trajectory, _ = self.get_trajectory(max_steps=max_steps)
            
            # prepare data for training
            states, actions, rewards = zip(*trajectory)
            states = torch.Tensor(states)
            actions = torch.Tensor(actions)
            
            # calculate expected discounted returns
            expected_returns_batch = list()
            for idx in range(len(rewards)):
                discounted_rewards = [self.gamma**i * reward for i, reward in enumerate(rewards[idx:])]
                expected_returns_batch.append(sum(discounted_rewards))
            
            # normalize and reformat expected returns
            expected_returns_batch = torch.FloatTensor(expected_returns_batch)
            expected_returns_batch /= expected_returns_batch.max()
            
            # calculate loss
            action_probs = self.policy_net(states)
            prob_batch = action_probs.gather(dim=1,index=actions.long().view(-1,1)).squeeze() 
            loss = self.criterion(prob_batch, expected_returns_batch)
            
            # optimize the model with backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def visualize_policy(self, num_episodes=1, max_steps=500):
        for _ in range(num_episodes):
            trajectory, done = self.get_trajectory(max_steps=max_steps, render=True)
            
    def evaluate_policy(self, num_episodes=100, max_steps=500):
        win = 0
        for _ in range(num_episodes):
            trajectory, done = self.get_trajectory(max_steps=max_steps)
            # CUSTOMIZE: for lunarlander environment
            if done and trajectory[-1][2] == 100:
                win += 1
        return win / num_episodes
