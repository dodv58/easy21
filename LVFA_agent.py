import numpy as np
import easy21_env as env
import random
import sys
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize
w = [0 for i in range(36)]
S = np.array(np.meshgrid(np.arange(1, 11), np.arange(1, 22))).T.reshape(-1, 2)
S = [tuple(item) for item in S]
A = [0,1]
gamma = 1

def get_features(s, a):
    tmp = [0 for i in range(36)]
    return tmp

def train(total_episodes=1000):
    history = []
    e = [0 for i in range(36)]
    episode = 0

    while episode < total_eposodes:
        current_state = env.start_game()
        current_action = get_action(current_state)
        is_terminated = False
        while not is_terminated:
            is_terminated, next_state, reward = env.step(current_state, current_action)
        

        episode += 1




def train(total_episodes = 1000, lamda = 0.1, true_Q = None):
    history = []
    state_visited_count = dict()
    s_a_selected_count = dict()
    def get_action(Q, s):
        # on-policy e-greedy
        epsilon = 10/(10 + state_visited_count[s])
        optimal_action = np.argmax(Q[s])
        return random.choices([optimal_action, 1-optimal_action], [1-epsilon+epsilon/2, epsilon/2])[0]

    Q = dict()
    for s in S:
        Q[s] = [0, 0]
        state_visited_count[s] = 0
        s_a_selected_count[s] = [0, 0]

    episodes = 0
    while episodes < total_episodes:
        E = dict()
        for s in S:
            E[s] = [0, 0]

        current_state = env.start_game() 
        current_action = get_action(Q, s)
        is_terminated = False

        while not is_terminated:
            is_terminated, next_state, reward = env.step(current_state, current_action)
            state_visited_count[current_state] += 1
            s_a_selected_count[current_state][current_action] += 1

            if not is_terminated:
                next_action = get_action(Q, next_state)
                delta = reward + gamma * Q[next_state][next_action] - Q[current_state][current_action]
            else:
                next_action = None
                delta = reward + gamma * 0 - Q[current_state][current_action]
            alpha = 1 / s_a_selected_count[current_state][current_action]
            E[current_state][current_action] = (1-alpha) * E[current_state][current_action] + 1
            for s in S:
                for a in A:
                    Q[s][a] += alpha * delta * E[s][a]
                    E[s][a] *= gamma * lamda

            current_state = next_state
            current_action = next_action


        episodes += 1
        if true_Q:
            history.append(math.sqrt(sum([(max(Q[s]) - true_Q[s])**2 for s in S])))
    return Q, history


df = pd.read_csv('out.csv')
true_Q = dict()
for item in df.itertuples(index=False):
    true_Q[(item[0],item[1])] = item[2]


for lamda in [0]:
    Q, history = train(10000, lamda, true_Q)
    mse = math.sqrt(sum([(max(Q[s]) - true_Q[s])**2 for s in S]))
    print(mse)

    x = np.arange(0, len(history))
    y = np.array(history)

# plotting
    plt.title("Line graph")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.plot(x, y, color ="red")
    plt.show()


