# ref: https://github.com/linesd/tabular-methods

from math import floor
import numpy as np
from gym import Env, spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# TODO: simplify this class
class GridWorld(Env):
    """
    Creates a gridworld object to pass to an RL algorithm.

    Parameters
    ----------
    num_rows : int
        The number of rows in the gridworld.

    num_cols : int
        The number of cols in the gridworld.

    start_state : numpy array of shape (1, 2), np.array([[row, col]])
        The start state of the gridworld (can only be one start state)

    goal_states : numpy arrany of shape (n, 2)
        The goal states for the gridworld where n is the number of goal
        states.
    """
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

    def __init__(self, num_rows, num_cols, start_state, goal_states):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.start_state = start_state
        self.goal_states = goal_states
        self.obs_states = None
        self.bad_states = None
        self.num_bad_states = 0
        self.p_good_trans = None
        self.bias = None
        self.r_step = None
        self.r_goal = None
        self.r_dead = None
        self.gamma = 1 # default is no discounting

    def add_obstructions(self, obstructed_states=None, bad_states=None, restart_states=None):
        """
        Add obstructions to the grid world.

        Obstructed states: walls that prohibit the agent from entering that state.

        Bad states: states that incur a greater penalty than a normal step.

        Restart states: states that incur a high penalty and transition the agent
                        back to the start state (but do not end the episode).

        Parameters
        ----------
        obstructed_states : numpy array of shape (n, 2)
            States the agent cannot enter where n is the number of obstructed states
            and the two columns are the row and col position of the obstructed state.

        bad_states: numpy array of shape (n, 2)
            States in which the agent incurs high penalty where n is the number of bad
            states and the two columns are the row and col position of the bad state.

        restart_states: numpy array of shape (n, 2)
            States in which the agent incurs high penalty and transitions to the start
            state where n is the number of restart states and the two columns are the
            row and col position of the restart state.
        """
        self.obs_states = obstructed_states
        self.bad_states = bad_states
        if bad_states is not None:
            self.num_bad_states = bad_states.shape[0]
        else:
            self.num_bad_states = 0
        self.restart_states = restart_states
        if restart_states is not None:
            self.num_restart_states = restart_states.shape[0]
        else:
            self.num_restart_states = 0

    def add_transition_probability(self, p_good_transition, bias):
        """
        Add transition probabilities to the grid world.

        p_good_transition is the probability that the agent successfully
        executes the intended action. The action is then incorrectly executed
        with probability 1 - p_good_transition and in tis case the agent
        transitions to the left of the intended transition with probability
        (1 - p_good_transition) * bias and to the right with probability
        (1 - p_good_transition) * (1 - bias).

        Parameters
        ----------
        p_good_transition : float (in the interval [0,1])
             The probability that the agents attempted transition is successful.

        bias : float (in the interval [0,1])
            The probability that the agent transitions left or right of the
            intended transition if the intended transition is not successful.
        """
        self.p_good_trans = p_good_transition
        self.bias = bias

    def add_rewards(self, step_reward, goal_reward, bad_state_reward=None, restart_state_reward = None):
        """
        Define which states incur which rewards.

        Parameters
        ----------
        step_reward : float
            The reward for each step taken by the agent in the grid world.
            Typically a negative value (e.g. -1).

        goal_reward : float
            The reward given to the agent for reaching the goal state.
            Typically a middle range positive value (e.g. 10)

        bad_state_reward : float
            The reward given to the agent for transitioning to a bad state.
            Typically a middle range negative value (e.g. -6)

        restart_state_reward : float
            The reward given to the agent for transitioning to a restart state.
            Typically a large negative value (e.g. -100)
        """
        self.r_step = step_reward
        self.r_goal = goal_reward
        self.r_bad = bad_state_reward
        self.r_restart = restart_state_reward

    def add_discount(self, discount):
        """
        Discount rewards so that recent rewards carry more weight than past rewards.

        Parameters
        ----------
        discount : float (in the interval [0, 1])
            The discount factor.
        """
        self.gamma = discount

    def create_gridworld(self):
        """
        Create the grid world with the specified parameters.

        Returns
        -------
        self : class object
            Holds information about the environment to solve
            such as the reward structure and the transition dynamics.
        """
        self.num_actions = 4  # [left, down, right, up]
        self.num_states = self.num_cols * self.num_rows
        self.observation_space = spaces.Discrete(self.num_states)
        self.action_space = spaces.Discrete(self.num_actions)

        self.start_state_seq = self._row_col_to_seq(self.start_state, self.num_cols)
        self.goal_states_seq = self._row_col_to_seq(self.goal_states, self.num_cols)
        self.obstruction_state_seqs = []
        self.bad_state_seqs = []
        self.restart_state_seqs = []
        if self.obs_states is not None:
            self.obstruction_state_seqs = self._row_col_to_seq(self.obs_states, self.num_cols)
        if self.bad_states is not None:
            self.bad_state_seqs = self._row_col_to_seq(self.bad_states, self.num_cols)
        if self.restart_states is not None:
            self.restart_state_seqs = self._row_col_to_seq(self.restart_states, self.num_cols)
        
        self.state = int(self.start_state_seq)

        # rewards structure
        self.R = self.r_step * np.ones((self.num_states, 1))
        # self.R[self.num_states-1] = 0
        self.R[self.goal_states_seq] = self.r_goal
        for i in range(self.num_bad_states):
            if self.r_bad is None:
                raise Exception("Bad state specified but no reward is given")
            bad_state = self._row_col_to_seq(self.bad_states[i,:].reshape(1,-1), self.num_cols)
            self.R[bad_state, :] = self.r_bad
        for i in range(self.num_restart_states):
            if self.r_restart is None:
                raise Exception("Restart state specified but no reward is given")
            restart_state = self._row_col_to_seq(self.restart_states[i,:].reshape(1,-1), self.num_cols)
            self.R[restart_state, :] = self.r_restart

        # probability model (transition function)
        if self.p_good_trans == None:
            raise Exception("Must assign probability and bias terms via the add_transition_probability method.")

        # TODO: add `p_good_trans` and `bias` to the transition function
        self.P = {s: {a: [0]*self.num_states for a in range(self.num_actions)} for s in range(self.num_states)}
        for state in range(self.num_states):
            for action in range(self.num_actions):
                if state in self.obstruction_state_seqs or state in self.goal_states_seq or state in self.restart_state_seqs:
                    self.P[state][action][state] = 1
                    continue

                row, col = self._seq_to_row_col(state, self.num_cols)[0]
                newrow, newcol = self._inc(row, col, action)
                seq = self._row_col_to_seq(np.array([[newrow, newcol]]), self.num_cols)[0]
                
                # check if the action move toward obstruction state
                if seq in self.obstruction_state_seqs:
                    self.P[state][action][state] = 1
                else:
                    self.P[state][action][seq] = 1

        """ old transition function
        self.P = np.zeros((self.num_states ,self.num_actions, self.num_states))

        for action in range(self.num_actions):
            for state in range(self.num_states):

                # check if state is the fictional end state - self transition
                if state == self.num_states-1:
                    self.P[state, state, action] = 1
                    continue

                # check if the state is the goal state or an obstructed state - transition to end
                row_col = self._seq_to_row_col(state, self.num_cols)
                if self.obs_states is not None:
                    end_states = np.vstack((self.obs_states, self.goal_states))
                else:
                    end_states = self.goal_states

                # check whether the state is an end (terminal) state
                if any(np.sum(np.abs(end_states-row_col), 1) == 0):
                    self.P[state, self.num_states-1, action] = 1

                # else consider stochastic effects of action
                else:
                    for dir in range(-1,2,1):
                        direction = self._get_direction(action, dir)
                        next_state = self._get_state(state, direction)
                        if dir == 0:
                            prob = self.p_good_trans
                        elif dir == -1:
                            prob = (1 - self.p_good_trans)*(self.bias)
                        elif dir == 1:
                            prob = (1 - self.p_good_trans)*(1-self.bias)

                        self.P[state, next_state, action] += prob

                # make restart states transition back to the start state with
                # probability 1
                if self.restart_states is not None:
                    if any(np.sum(np.abs(self.restart_states-row_col),1)==0):
                        next_state = self._row_col_to_seq(self.start_state, self.num_cols)
                        self.P[state,:,:] = 0
                        self.P[state,next_state,:] = 1
        """
        
        return self

    def reset(self):
        self.state = int(self.start_state_seq)
        return int(self.start_state_seq)
    
    def step(self, action):
        probs = self.P[self.state][action]
        new_state = np.random.choice(self.num_states, p=probs)
        self.state = new_state

        done = False
        if new_state in self.goal_states_seq or new_state in self.restart_state_seqs:
            done = True
        
        return new_state, self.R[new_state], done, {"probs": probs}

    def render(self, value_function=None, policy=None, state_counts=None, title=None, path=None):
        """
        Plots the grid world solution.

        Parameters
        ----------
        model : python object
            Holds information about the environment to solve
            such as the reward structure and the transition dynamics.

        value_function : numpy array of shape (N, 1)
            Value function of the environment where N is the number
            of states in the environment.

        policy : numpy array of shape (N, 1)
            Optimal policy of the environment.

        title : string
            Title of the plot. Defaults to None.

        path : string
            Path to save image. Defaults to None.
        """

        if value_function is not None and state_counts is not None:
            raise Exception("Must supple either value function or state_counts, not both!")

        fig, ax = plt.subplots()

        # add features to grid world
        if value_function is not None:
            self._add_value_function(value_function, "Value function")
        elif state_counts is not None:
            self._add_value_function(state_counts, "State counts")
        elif value_function is None and state_counts is None:
            self._add_value_function(value_function, "Value function")

        self._add_patches(ax)
        self._add_policy(policy)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                fancybox=True, shadow=True, ncol=3)
        if title is not None:
            plt.title(title, fontdict=None, loc='center')
        if path is not None:
            plt.savefig(path, dpi=300, bbox_inches='tight')

        plt.show()

    def _inc(self, row, col, a):
        if a == self.LEFT:
            col = max(col - 1, 0)
        elif a == self.DOWN:
            row = min(row + 1, self.num_rows - 1)
        elif a == self.RIGHT:
            col = min(col + 1, self.num_cols - 1)
        elif a == self.UP:
            row = max(row - 1, 0)
        return (row, col)

    def _get_direction(self, action, direction):
        """
        Takes is a direction and an action and returns a new direction.

        Parameters
        ----------
        action : int
            The current action 0, 1, 2, 3 for gridworld.

        direction : int
            Either -1, 0, 1.

        Returns
        -------
        direction : int
            Value either 0, 1, 2, 3.
        """
        left = [2,3,1,0]
        right = [3,2,0,1]
        if direction == 0:
            new_direction = action
        elif direction == -1:
            new_direction = left[action]
        elif direction == 1:
            new_direction = right[action]
        else:
            raise Exception("getDir received an unspecified case")
        return new_direction

    def _get_state(self, state, direction):
        """
        Get the next_state from the current state and a direction.

        Parameters
        ----------
        state : int
            The current state.

        direction : int
            The current direction.

        Returns
        -------
        next_state : int
            The next state given the current state and direction.
        """
        row_change = [-1,1,0,0]
        col_change = [0,0,-1,1]
        row_col = self._seq_to_row_col(state, self.num_cols)
        row_col[0,0] += row_change[direction]
        row_col[0,1] += col_change[direction]

        # check for invalid states
        if self.obs_states is not None:
            if (np.any(row_col < 0) or
                np.any(row_col[:,0] > self.num_rows-1) or
                np.any(row_col[:,1] > self.num_cols-1) or
                np.any(np.sum(abs(self.obs_states - row_col), 1)==0)):
                next_state = state
            else:
                next_state = self._row_col_to_seq(row_col, self.num_cols)[0]
        else:
            if (np.any(row_col < 0) or
                np.any(row_col[:,0] > self.num_rows-1) or
                np.any(row_col[:,1] > self.num_cols-1)):
                next_state = state
            else:
                next_state = self._row_col_to_seq(row_col, self.num_cols)[0]

        return next_state

    def _row_col_to_seq(self, row_cols, num_cols):
        return row_cols[:,0] * num_cols + row_cols[:,1]

    def _seq_to_row_col(self, seq, num_cols):
        r = floor(seq / num_cols)
        c = seq - r * num_cols
        return np.array([[r, c]])

    def _add_value_function(self, value_function, name):

        if value_function is not None:
            # colobar max and min
            vmin = np.min(value_function)
            vmax = np.max(value_function)
            # reshape and set obstructed states to low value
            val = value_function[:, 0].reshape(self.num_rows, self.num_cols)
            if self.obs_states is not None:
                index = self.obs_states
                val[index[:, 0], index[:, 1]] = -100
            plt.imshow(val, vmin=vmin, vmax=vmax, zorder=0)
            plt.colorbar(label=name)
        else:
            val = np.zeros((self.num_rows, self.num_cols))
            plt.imshow(val, zorder=0)
            plt.yticks(np.arange(-0.5, self.num_rows+0.5, step=1))
            plt.xticks(np.arange(-0.5, self.num_cols+0.5, step=1))
            plt.grid()
            plt.colorbar(label=name)

    def _add_patches(self, ax):

        start = patches.Circle(tuple(np.flip(self.start_state[0])), 0.2, linewidth=1,
                            edgecolor='b', facecolor='b', zorder=1, label="Start")
        ax.add_patch(start)

        for i in range(self.goal_states.shape[0]):
            end = patches.RegularPolygon(tuple(np.flip(self.goal_states[i, :])), numVertices=5,
                                        radius=0.25, orientation=np.pi, edgecolor='g', zorder=1,
                                        facecolor='g', label="Goal" if i == 0 else None)
            ax.add_patch(end)

        # obstructed states patches
        if self.obs_states is not None:
            for i in range(self.obs_states.shape[0]):
                obstructed = patches.Rectangle(tuple(np.flip(self.obs_states[i, :]) - 0.35), 0.7, 0.7,
                                            linewidth=1, edgecolor='gray', facecolor='black', zorder=1,
                                            label="Obstructed" if i == 0 else None)
                ax.add_patch(obstructed)

        if self.bad_states is not None:
            for i in range(self.bad_states.shape[0]):
                bad = patches.Wedge(tuple(np.flip(self.bad_states[i, :])), 0.2, 40, -40,
                                    linewidth=1, edgecolor='gray', facecolor='orange', zorder=1,
                                    label="Bad state" if i == 0 else None)
                ax.add_patch(bad)

        if self.restart_states is not None:
            for i in range(self.restart_states.shape[0]):
                restart = patches.Wedge(tuple(np.flip(self.restart_states[i, :])), 0.2, 40, -40,
                                        linewidth=1, edgecolor='gray', facecolor='r', zorder=1,
                                        label="Restart state" if i == 0 else None)
                ax.add_patch(restart)

    def _add_policy(self, policy):

        if policy is not None:
            # define the gridworld
            X = np.arange(0, self.num_cols, 1)
            Y = np.arange(0, self.num_rows, 1)

            # define the policy direction arrows
            U, V = self._create_policy_direction_arrays(policy)
            # remove the obstructions and final state arrows
            ra = self.goal_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan
            if self.obs_states is not None:
                ra = self.obs_states
                U[ra[:, 0], ra[:, 1]] = np.nan
                V[ra[:, 0], ra[:, 1]] = np.nan
            if self.restart_states is not None:
                ra = self.restart_states
                U[ra[:, 0], ra[:, 1]] = np.nan
                V[ra[:, 0], ra[:, 1]] = np.nan

            plt.quiver(X, Y, U, V, zorder=10, label="Policy")

    def _create_policy_direction_arrays(self, policy):
        """
        define the policy directions
        0 - left  [-1, 0]
        1 - down  [0, -1]
        2 - right [1, 0]
        3 - up    [0, 1]
        :param policy:
        :return:
        """

        # intitialize direction arrays
        U = np.zeros((self.num_rows, self.num_cols))
        V = np.zeros((self.num_rows, self.num_cols))

        for state in range(self.num_states-1):
            # get index of the state
            i = tuple(self._seq_to_row_col(state, self.num_cols)[0])
            # define the arrow direction
            if policy[state] == self.UP:
                U[i] = 0
                V[i] = 0.5
            elif policy[state] == self.DOWN:
                U[i] = 0
                V[i] = -0.5
            elif policy[state] == self.LEFT:
                U[i] = -0.5
                V[i] = 0
            elif policy[state] == self.RIGHT:
                U[i] = 0.5
                V[i] = 0

        return U, V


if __name__ == '__main__':
    # specify world parameters
    num_cols = 10
    num_rows = 10
    obstructions = np.array([[0,7],[1,1],[1,2],[1,3],[1,7],[2,1],[2,3],
                            [2,7],[3,1],[3,3],[3,5],[4,3],[4,5],[4,7],
                            [5,3],[5,7],[5,9],[6,3],[6,9],[7,1],[7,6],
                            [7,7],[7,8],[7,9],[8,1],[8,5],[8,6],[9,1]])
    bad_states = np.array([[1,9],[4,2],[4,4],[7,5],[9,9]])
    restart_states = np.array([[3,7],[8,2]])
    start_state = np.array([[0,4]])
    goal_states = np.array([[0,9],[2,2],[8,7]])

    # create model
    gw = GridWorld(num_rows=num_rows,
                num_cols=num_cols,
                start_state=start_state,
                goal_states=goal_states)
    gw.add_obstructions(obstructed_states=obstructions,
                        bad_states=bad_states,
                        restart_states=restart_states)
    gw.add_rewards(step_reward=-1,
                goal_reward=10,
                bad_state_reward=-6,
                restart_state_reward=-10)
    gw.add_transition_probability(p_good_transition=1.0, bias=0)
    gw.add_discount(discount=0.9)
    model = gw.create_gridworld()

    # plot the environment
    model.plot_gridworld(title="Sample Gridworld")
