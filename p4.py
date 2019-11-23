import numpy as np
from hiive.visualization import mdpviz
import gym
import os
import sys
import matplotlib.pyplot as plt
import mdptoolbox
import mdptoolbox.example
import time

import matplotlib

import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import itertools
matplotlib.style.use('ggplot')

#### PLOTING #####
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    #plt.show()
    plt.savefig(env.name)


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        #plt.show()
        plt.savefig(env.name)

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))



def plot_episode_stats(stats, smoothing_window=10, noshow=False, plt_name="default"):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        #plt.show(fig1)
        plt.savefig(plt_name + "_elt.png")

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        #plt.show(fig2)
        plt.savefig(plt_name + "_er.png")

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        #plt.show(fig3)
        plt.savefig(plt_name + "ts_en.png")

    return fig1, fig2, fig3
##### PLOTING END ########

class FireManagementSpec():
    def __init__(self, population_classes=7, fire_classes=13, seed=1234, verbose=True):
        self.seed = seed
        self.verbose = verbose
        self.population_classes = population_classes
        self.fire_classes = fire_classes
        self.states = {}

        self._action_do_nothing = None
        self._action_burn = None

        self._probabilities = {}
        self.name=f'fire_management_{population_classes}_{fire_classes}_{seed}'
        self.n_actions=2
        self.n_states=self.fire_classes * self.population_classes

        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self._setup_mdp()

    def _reset_state_probabilities(self):
        self._probabilities = {}

    def _get_probability_for_state(self, pc, fc):
        state_name = self._get_state_name(pc, fc)
        if state_name not in self._probabilities:
            return None
        return self._probabilities[state_name]

    def _set_probability_for_state(self, pc, fc, p):
        state_name = self._get_state_name(pc, fc)
        if state_name not in self._probabilities:
            self._probabilities[state_name] = 0.
        self._probabilities[state_name] += p
        return self._probabilities[state_name]

    @staticmethod
    def _is_terminal(s):
        return False  # s == 0

    @staticmethod
    def get_habitat_suitability(years):
        if years < 0:
            msg = "Invalid years '%s', it should be positive." % str(years)
            raise ValueError(msg)
        if years <= 5:
            return 0.2 * years
        elif 5 <= years <= 10:
            return -0.1 * years + 1.5
        else:
            return 0.5

    @staticmethod
    def _get_state_name(pc, fc):
        return f'pc:{pc}, fc:{fc}'

    def _get_state(self, pc, fc):
        state_name = self._get_state_name(pc, fc)
        is_terminal = self._is_terminal(pc)
        if state_name not in self.states:
            state = self.spec.state(name=state_name, terminal_state=is_terminal)
            self.states[state_name] = state
        # print(f'{state_name} : {is_terminal}')
        state = self.states[state_name]
        return state

    def _add_state_transition_and_reward(self, pc, fc, action):
        cs = self._get_state(pc, fc)
        results = self._get_reward_and_new_state_values(pc, fc, action)
        for reward, npc, nfc, tp in results:
            ns = self._get_state(npc, nfc)
            ns = mdpviz.NextState(state=ns, weight=tp)
            self.spec.transition(state=cs, action=action, outcome=ns)
            self.spec.transition(state=cs, action=action, outcome=mdpviz.Reward(reward))
            if self.verbose:
                print(f'[state:action]: [{(pc, fc)} : {action.name}] -> new state: {(npc, nfc)}, '
                      f'p(t): {tp}, reward: {reward} ')

    def transition_fire_class(self, fc, action):
        if action == self._action_do_nothing:
            return (fc + 1) if fc < self.fire_classes - 1 else fc
        elif action == self._action_burn:
            return 0
        return fc

    def _get_reward_and_new_state_values(self, pc, fc, action, default_p=0.5):
        pop_change_down = -1
        pop_change_same = 0

        self._probabilities = {}

        r = self.get_habitat_suitability(fc)
        fc = self.transition_fire_class(fc, action)
        if pc == 0:
            # dead
            return [[0.0, 0, fc, 1.0]]  # stays in same state
        if pc == self.population_classes - 1:
            pop_change_up = 0
            if action == self._action_burn:
                pop_change_same -= 1
                pop_change_down -= 1

            tsd = self._set_probability_for_state(pc=pc + pop_change_down,
                                                  fc=fc,
                                                  p=(1.0 - default_p) * (1.0 - r))
            tss = self._set_probability_for_state(pc=pc + pop_change_same,
                                                  fc=fc,
                                                  p=1.0 - tsd)
        else:
            # Population abundance class can stay the same, transition up, or
            # transition down.
            pop_change_same = 0
            pop_change_up = 1
            pop_change_down = -1

            # If action 1 is taken, then the patch is burned so the population
            # abundance moves down a class.
            if action == self._action_burn:
                pop_change_same -= 1
                pop_change_up -= 1
                pop_change_down -= (1 if pop_change_down > 0 else 0)

            tss = self._set_probability_for_state(pc=pc + pop_change_same,
                                                  fc=fc,
                                                  p=default_p)

            tsu = self._set_probability_for_state(pc=pc + pop_change_up,
                                                  fc=fc,
                                                  p=(1 - default_p)*r)
            # In the case when transition_down = 0 before the effect of an action
            # is applied, then the final state is going to be the same as that for
            # transition_same, so we need to add the probabilities together.
            tsd = self._set_probability_for_state(pc=pc + pop_change_down,
                                                  fc=fc,
                                                  p=(1 - default_p)*(1 - r))

        # build results
        results = []

        npc_up = pc + pop_change_up
        npc_down = pc + pop_change_down
        npc_same = pc + pop_change_same

        transition_probabilities = {
            (npc_up, self._get_probability_for_state(npc_up, fc)),
            (npc_down, self._get_probability_for_state(npc_down, fc)),
            (npc_same, self._get_probability_for_state(npc_same, fc))
        }

        for npc, probability in transition_probabilities:
            if probability is not None and probability > 0.0:
                reward = int(pc > 0)
                results.append((reward, npc, fc, probability))

        return results

    def _setup_mdp(self):
        self.spec = mdpviz.MDPSpec()

        self._action_do_nothing = self.spec.action('do_nothing')
        self._action_burn = self.spec.action('burn')

        # build transitions
        for pc in range(0, self.population_classes):
            if self._is_terminal(pc):
                continue
            for fc in range(0, self.fire_classes):
                # actions
                self._add_state_transition_and_reward(pc=pc, fc=fc, action=self._action_do_nothing)
                self._add_state_transition_and_reward(pc=pc, fc=fc, action=self._action_burn)
                if self.verbose:
                    print()

    def get_transition_and_reward_objects(self, p_default=0.5):
        return self.spec.get_transition_and_reward_arrays(p_default)

    def to_graph(self):
        return self.spec.to_graph()

    def to_env(self):
        return self.spec.to_discrete_env()


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap

def evaluate_rewards_and_transitions(problem, mutate=False):
    # Enumerate state and action space sizes
    num_states = problem.observation_space.n
    num_actions = problem.action_space.n

    # Intiailize T and R matrices
    R = np.zeros((num_states, num_actions, num_states))
    T = np.zeros((num_states, num_actions, num_states))

    # Iterate over states, actions, and transitions
    for state in range(num_states):
        for action in range(num_actions):
            for transition in problem.env.P[state][action]:
                probability, next_state, reward, done = transition
                R[state, action, next_state] = reward
                T[state, action, next_state] = probability

            # Normalize T across state + action axes
            T[state, action, :] /= np.sum(T[state, action, :])

    # Conditionally mutate and return
    if mutate:
        problem.env.R = R
        problem.env.T = T
    return R, T

@timing
def value_iteration(problem, R=None, T=None, gamma=0.9, max_iterations=10**6, delta=10**-3):
    """ Runs Value Iteration on a gym problem """
    value_fn = np.zeros(problem.observation_space.n)
    if R is None or T is None:
        R, T = evaluate_rewards_and_transitions(problem)

    for i in range(max_iterations):
        previous_value_fn = value_fn.copy()
        Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
        value_fn = np.max(Q, axis=1)

        if np.max(np.abs(value_fn - previous_value_fn)) < delta:
            break

    # Get and return optimal policy
    policy = np.argmax(Q, axis=1)
    return policy, i + 1

def encode_policy(policy, shape):
    """ One-hot encodes a policy """
    encoded_policy = np.zeros(shape)
    encoded_policy[np.arange(shape[0]), policy] = 1
    return encoded_policy

@timing
def policy_iteration(problem, R=None, T=None, gamma=0.9, max_iterations=10**6, delta=10**-3):
    """ Runs Policy Iteration on a gym problem """
    num_spaces = problem.observation_space.n
    num_actions = problem.action_space.n

    # Initialize with a random policy and initial value function
    policy = np.array([problem.action_space.sample() for _ in range(num_spaces)])
    value_fn = np.zeros(num_spaces)

    # Get transitions and rewards
    if R is None or T is None:
        R, T = evaluate_rewards_and_transitions(problem)
    
    #pi = mdptoolbox.mdp.PolicyIterationModified(R, T, 0.9)
    # Iterate and improve policies
    for i in range(max_iterations):
        previous_policy = policy.copy()

        for j in range(max_iterations):
            previous_value_fn = value_fn.copy()
            Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
            value_fn = np.sum(encode_policy(policy, (num_spaces, num_actions)) * Q, 1)

            if np.max(np.abs(previous_value_fn - value_fn)) < delta:
                break

        Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
        policy = np.argmax(Q, axis=1)

        if np.array_equal(policy, previous_policy):
            break

    # Return optimal policy
    return policy, i + 1

def print_policy(policy, mapping=None, shape=(0,)):
    print(np.array([mapping[action] for action in policy]).reshape(shape))

def run_discrete(environment_name, mapping, shape=None, gym_env_name=True):
    if(gym_env_name):
        problem = gym.make(environment_name)
    else:
        problem = environment_name

    print(environment_name + "#"*45)
    print('Actions:', problem.env.action_space.n)
    print('States:', problem.env.observation_space.n)
    print(problem.env.desc)
    print()

    print("value iteration")
    value_policy, iters = value_iteration(problem)
    print('Iterations:', iters)
    print()

    print("policy iteration")
    policy, iters = policy_iteration(problem)
    print('iter:', iters)
    print()

    diff = sum([abs(x-y) for x, y in zip(policy.flatten(), value_policy.flatten())])
    if diff > 0:
        print('diff:', diff)
        print()

    if shape is not None:
        print("policy")
        print_policy(policy, mapping, shape)
        print_policy(value_policy, mapping, shape)
        print()

    return policy

def run_discrete_fire(env, mapping, shape=None):

    print("Fire " + "#"*45)
    print('Actions:', env.action_space.n)
    print('States:', env.observation_space.n)
    #print(problem.env.desc)
    print()

    print("value iteration")
    value_policy, iters = value_iteration_fire(env)
    print('Iterations:', iters)
    print()

    print("policy iteration")
    policy, iters = policy_iteration_fire(env)
    print('iter:', iters)
    print()

    diff = sum([abs(x-y) for x, y in zip(policy.flatten(), value_policy.flatten())])
    if diff > 0:
        print('diff:', diff)
        print()

    if shape is not None:
        print("policy")
        print_policy(policy, mapping, shape)
        print_policy(value_policy, mapping, shape)
        print()

    return policy

@timing
def value_iteration_fire(problem, R=None, T=None, gamma=0.9, max_iterations=10**6, delta=10**-3):
    """ Runs Value Iteration on a gym problem """
    value_fn = np.zeros(problem.observation_space.n)
    if R is None or T is None:
        R, T = evaluate_rewards_and_transitions_fire(problem)

    for i in range(max_iterations):
        previous_value_fn = value_fn.copy()
        Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
        value_fn = np.max(Q, axis=1)

        if np.max(np.abs(value_fn - previous_value_fn)) < delta:
            break

    # Get and return optimal policy
    policy = np.argmax(Q, axis=1)
    return policy, i + 1

@timing
def policy_iteration_fire(problem, R=None, T=None, gamma=0.9, max_iterations=10**6, delta=10**-3):
    """ Runs Policy Iteration on a gym problem """
    num_spaces = problem.observation_space.n
    num_actions = problem.action_space.n

    # Initialize with a random policy and initial value function
    policy = np.array([problem.action_space.sample() for _ in range(num_spaces)])
    value_fn = np.zeros(num_spaces)

    # Get transitions and rewards
    if R is None or T is None:
        R, T = evaluate_rewards_and_transitions_fire(problem)
            
    #pi = mdptoolbox.mdp.PolicyIterationModified(R, T, 0.9)
    # Iterate and improve policies
    for i in range(max_iterations):
        previous_policy = policy.copy()

        for j in range(max_iterations):
            previous_value_fn = value_fn.copy()
            Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
            value_fn = np.sum(encode_policy(policy, (num_spaces, num_actions)) * Q, 1)

            if np.max(np.abs(previous_value_fn - value_fn)) < delta:
                break

        Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
        policy = np.argmax(Q, axis=1)

        if np.array_equal(policy, previous_policy):
            break

    # Return optimal policy
    return policy, i + 1

def evaluate_rewards_and_transitions_fire(env, mutate=False):
    # Enumerate state and action space sizes
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Intiailize T and R matrices
    R = np.zeros((num_states, num_actions, num_states))
    T = np.zeros((num_states, num_actions, num_states))

    # Iterate over states, actions, and transitions
    for state in range(num_states):
        for action in range(num_actions):
            for transition in env.P[state][action]:
                probability, next_state, reward, done = transition
                R[state, action, next_state] = reward
                T[state, action, next_state] = probability

            # Normalize T across state + action axes
            T[state, action, :] /= np.sum(T[state, action, :])

    # Conditionally mutate and return
    if mutate:
        problem.env.R = R
        problem.env.T = T
    return R, T

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            #print(t)
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            ##max amount of iterations
            if(t > 1000):
                done = True

            if done:
                break
                
            state = next_state
    
    return Q, stats

# # # FROZEN LAKE SMALL
# mapping = {0: "L", 1: "D", 2: "R", 3: "U"}
# # shape = (4, 4)
# # run_discrete('FrozenLake-v0', mapping, shape)

# # FROZEN LAKE LARGE
# shape = (8, 8)
# run_discrete('FrozenLake8x8-v0', mapping ,shape)

# PI/VI TAXI and FIRE MANAGEMENT
#####TAXI#####
mapping = {0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
run_discrete('Taxi-v3', mapping)
print("#"*30)

    
fm_spec = FireManagementSpec(seed=6094, population_classes=7, fire_classes=13, verbose=False)
# # get env for gym
fm_env = fm_spec.to_env()
#####FIRE#######
r = run_discrete_fire(fm_env, {0:"noting", 1:"cut"}, (13,7))

t,r = fm_spec.get_transition_and_reward_objects()
vi = mdptoolbox.mdp.ValueIteration(t, r, 0.9)
t1 = time.time()
vi.run()
t2 = time.time()
print("VI run time: ", vi.time* 1000.0)
# print(vi.P)
print(vi.P)
print("vi policy", vi.policy) # result is (0, 0, 0)
print("iter", vi.iter)
print("#"*30)
print()

t,r = fm_spec.get_transition_and_reward_objects()
pi = mdptoolbox.mdp.PolicyIterationModified(t, r, 0.9)
t1 = time.time()
pi.run()
t2 = time.time()

print("PI run time: ", pi.time* 1000.0)
print("pi policy", pi.policy) # result is (0, 0, 0)
print("iter", pi.iter)
print("#"*30)
print()

# ##### Q - LEARNING PART
env = gym.make('Taxi-v3')
Q, stats = q_learning(env, 500)
plot_episode_stats(stats=stats, plt_name="taxi_v3")
print("#"*30)
print()

fm_spec = FireManagementSpec(seed=694, population_classes=3, fire_classes=7, verbose=False)
fm_env = fm_spec.to_env()

Q, stats = q_learning(fm_env, 500, discount_factor=.9, alpha=0.1, epsilon=0.41)
plot_episode_stats(stats=stats, plt_name="fire_management")
q = mdptoolbox.mdp.QLearning(t, r, .9)

t1 = time.time()
q.run()
t2 = time.time()

print("Q run time: ", q.time* 1000.0)
# print(pi.P)
# print(pi.R)
print("Q policy", q.policy) # result is (0, 0, 0)
print()

