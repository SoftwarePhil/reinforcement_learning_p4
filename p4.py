import numpy as np
from hiive.visualization import mdpviz
import gym
import os

import matplotlib.pyplot as plt
import mdptoolbox
import mdptoolbox.example
import time

import matplotlib

import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class FireManagementSpec:
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
    
fm_spec = FireManagementSpec(seed=694, verbose=False)
# get env for gym
fm_env = fm_spec.to_env()
#####FIRE#######
run_discrete_fire(fm_env, {})
