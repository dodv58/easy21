import numpy as np
import easy21_env as env
import random
import sys
import pandas as pd

S = np.array(np.meshgrid(np.arange(1, 11), np.arange(1, 22))).T.reshape(-1, 2)
S = [tuple(item) for item in S]
A = [0,1]

def index_of_state(s):
    try:
        return S.index(s)
    except:
        print(s)
        sys.exit('Dead!!!')

# Initialize
df = pd.read_csv('out.csv')
Policies = [0 for i in range(len(S))]
for idx, row in df.iterrows():
    Policies[index_of_state((row['dealer_show'], row['player_sum']))] = row['action']


def get_action(s):
    # on-policy
    state_idx = index_of_state(s)
    return Policies[state_idx]

episode = 0
wins = 0
loses = 0
draws = 0

while episode < 10000:
    current_state = env.start_game()
    current_action = get_action(current_state)
    is_terminated = False
    while not is_terminated:
        is_terminated, next_state, reward = env.step(current_state, current_action)
        current_state = next_state
        current_action = get_action(current_state) if not is_terminated else -1

    if reward == -1:
        loses += 1
    elif reward == 1:
        wins += 1
    else:
        draws += 1
    episode += 1

print("{} - {} - {}".format(wins, draws, loses))

