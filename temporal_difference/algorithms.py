""" some function are customized for the CliffWalking-v0 environment """

import numpy as np
import pandas as pd


class SARSA:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # initialize the Q-table with zeros
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        
        # CUSTOMIZE: the environment shape for CliffWalking-v0 is (4, 12)
        self.env_shape = (4, 12)

    def choose_action(self, state):
        raise NotImplementedError()

    def update_Q(self, states, actions, rewards):
        raise NotImplementedError()

    def generate_sample_episode(self):
        raise NotImplementedError()

    def train(self, episodes):
        raise NotImplementedError()
    
    def evaluate_policy(self, episodes):
        raise NotImplementedError()
    
    def visualize_q_table(self):
        raise NotImplementedError()


class NStepSARSA(SARSA):
    def __init__(self, env, n_step=1, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(env, alpha, gamma, epsilon)
        self.n_step = n_step

    def choose_action(self, state, always_greedy=False):
        """
        Chooses an action using an epsilon-greedy policy.
        """
        if always_greedy:
            return np.argmax(self.Q[state])
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def update_Q(self, states, actions, rewards):
        """
        Updates the Q-table using N-Step SARSA.
        """
        G = 0
        for i in range(len(rewards)):
            G += (self.gamma ** i) * rewards[i]

        # Update Q(st,at):= Q(st,at) + lr [[G(t:t+n) + (gamma ** n_step) * Q(s(t+n),a(t+n))] - Q(st,at)]
        state, action = states[0], actions[0]
        self.Q[state, action] += self.alpha * (G + (self.gamma ** self.n_step) * self.Q[states[-1], actions[-1]] - self.Q[state, action])

    # TODO: add display option for visualization of the episode
    def generate_sample_episode(self, display=False, always_greedy=False):
        episode = list()
        off_cliff_info = {"status": False, "prev_state": None}
    
        state = self.env.reset()
        while True:
            timestep = list()
            prev_state = state
            
            action = self.choose_action(state, always_greedy=always_greedy)
            state, reward, done, _ = self.env.step(action)
            
            timestep.append(prev_state)
            timestep.append(action)
            timestep.append(reward)
            episode.append(timestep)

            # CUSTOMIZE: if the agent falls off the cliff, the episode ends
            if reward == -100:
                done = True
                off_cliff_info["status"] = True
                off_cliff_info["prev_state"] = prev_state

            if done:
                # include the terminal state
                episode.append([state, None, None])
                break
        
        return episode, off_cliff_info

    def train(self, episodes):
        """
        Trains the agent using N-Step SARSA.
        """
        for _ in range(episodes):
            state = self.env.reset()

            states = [state]
            actions = [self.choose_action(state)]
            rewards = [0.0]

            t = 0
            T = float("inf")
            while True:
                if t < T:
                    next_state, reward, done, _ = self.env.step(actions[-1])

                    states.append(next_state)
                    rewards.append(reward)

                    if done:
                        T = t + 1
                    else:
                        next_action = self.choose_action(next_state)
                        actions.append(next_action)

                tau = t - self.n_step + 1

                if tau >= 0:
                    self.update_Q(states[tau:tau+self.n_step+1], actions[tau:tau+self.n_step], rewards[tau:tau+self.n_step])

                if tau == T - 1:
                    break

                t += 1

    def evaluate_policy(self, episodes, display=False):
        wins = 0
        for _ in range(episodes):
            episode, off_cliff_info = self.generate_sample_episode(display=False)
            if not off_cliff_info["status"]:
                wins += 1
        return wins / episodes
    
    def visualize_q_table(self, colored_states=[]):
        q_table = np.zeros(self.env_shape + (self.env.action_space.n,))
    
        for state in range(self.env.observation_space.n):
            row = state // self.env_shape[1]
            col = state % self.env_shape[1]
            q_table[row, col] = self.Q[state]

        # calculate the average q-values for each state
        avg_q_values = np.mean(q_table, axis=2)
        df = pd.DataFrame(avg_q_values)
        
        return df.style.apply(self._style_trajectory_cells, axis=None, colored_states=colored_states)
    
    def _get_row_col(self, state):
        row = state // self.env_shape[1]
        col = state % self.env_shape[1]
        return row, col

    # CUSTOMIZE: for cliffwalking environment
    def _style_trajectory_cells(self, x, colored_states):
        df_colored = pd.DataFrame('background-color: black', index=x.index, columns=x.columns)
        
        df_colored.iloc[3, 0] = 'background-color: blue'
        df_colored.iloc[3, 11] = 'background-color: darkgreen'
        df_colored.iloc[3, 1:11] = 'background-color: brown'
        
        for state in colored_states:
            row, col = self._get_row_col(state)
            df_colored.iloc[row, col] = 'background-color: darkblue'
        
        return df_colored


class QLearning(NStepSARSA):
    n_step = 1

    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(env, self.n_step, alpha, gamma, epsilon)

    def update_Q(self, states, actions, rewards):
        """
        Updates the Q-table using Q-Learning.
        """
        state, action, reward = states[0], actions[0], rewards[0]
        next_state = states[-1]
        
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        max_next_Q = np.max(self.Q[next_state])
        target = reward + self.gamma * max_next_Q
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        
    def visualize_q_table(self, colored_states=[]):
        q_table = np.zeros(self.env_shape + (self.env.action_space.n,))
    
        for state in range(self.env.observation_space.n):
            row = state // self.env_shape[1]
            col = state % self.env_shape[1]
            q_table[row, col] = self.Q[state]

        # select highest Q-value from each state
        max_q_values = np.max(q_table, axis=2)
        df = pd.DataFrame(max_q_values)

        return df.style.apply(self._style_trajectory_cells, axis=None, colored_states=colored_states)


class ExpectedSARSA(NStepSARSA):
    n_step = 1

    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(env, self.n_step, alpha, gamma, epsilon)

    def update_Q(self, states, actions, rewards):
        """
        Updates the Q-table using Expected SARSA.
        """
        state, action, reward = states[0], actions[0], rewards[0]
        next_state = states[-1]

        # calculate action probabilities based on  epsilon-greedy policy
        action_probs = np.ones(self.env.action_space.n) * self.epsilon / self.env.action_space.n
        action_probs[np.argmax(self.Q[next_state])] += 1 - self.epsilon

        # calculate the expected Q-value of the next state-action pair (apply epsilon-greedy adjustment on the action probabilities)
        expected_next_Q = np.sum(self.Q[next_state] * action_probs)

        # update Q-Table
        target = reward + self.gamma * expected_next_Q
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def visualize_q_table(self, colored_states=[]):
        q_table = np.zeros(self.env_shape + (self.env.action_space.n,))
    
        # calculate weighted average Q-value from each state and assign to the new Q-table for visualization
        for state in range(self.env.observation_space.n):
            row = state // self.env_shape[1]
            col = state % self.env_shape[1]
            
            # calculate action probabilities based on  epsilon-greedy policy
            action_probs = np.ones(self.env.action_space.n) * self.epsilon / self.env.action_space.n
            action_probs[np.argmax(self.Q[state])] += 1 - self.epsilon
            weighted_q_value = np.sum(self.Q[state] * action_probs)
            q_table[row, col] = [weighted_q_value] * self.env.action_space.n

        # reshape the Q-table to 2D for visualization
        weighted_q_values = np.mean(q_table, axis=2)
        df = pd.DataFrame(weighted_q_values)

        return df.style.apply(self._style_trajectory_cells, axis=None, colored_states=colored_states)
