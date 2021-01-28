import numpy as np
import easy21_env as env
import random
import sys
import pandas as pd

S = np.array(np.meshgrid(np.arange(1, 11), np.arange(1, 22))).T.reshape(-1, 2)
S = [tuple(item) for item in S]
A = [0,1]

# Initialize
Q = [[0 for j in range(len(A))] for i in range(len(S))]
Policies = [[0.5 for j in range(len(A))] for i in range(len(S))]
N = {}
state_visited_count = {}
for item in S:
    state_visited_count[item] = 0

def index_of_state(s):
    try:
        return S.index(s)
    except:
        print(s)
        sys.exit('Dead!!!')

def get_action(s):
    # on-policy
    state_idx = index_of_state(s)
    return random.choices(A, Policies[state_idx])[0]

def update_policy(s):
    state_idx = index_of_state(s)
    idx = np.argmax(Q[state_idx])
    epsilon = 100/(100 + state_visited_count[s])
    Policies[state_idx][idx] = 1 - epsilon + epsilon/len(A)
    Policies[state_idx][1-idx] = epsilon/len(A)


episode = 0
results = 0

while episode < 2000000:
    current_state = env.start_game()
    current_action = get_action(current_state)
    is_terminated = False
    chain = []
    G = []
    while not is_terminated:
        is_terminated, next_state, reward = env.step(current_state, current_action)
        chain.append((current_state, current_action, reward))
        state_visited_count[current_state] += 1 
        current_state = next_state
        current_action = get_action(current_state) if not is_terminated else -1
    results += reward

    for i, block in enumerate(chain):
        G.append(sum([item[2] for item in chain[i:]]))
        state_idx = index_of_state(block[0])

        if (block[0], block[1]) not in N:
            N[(block[0], block[1])] = 1
        else:
            N[(block[0], block[1])] += 1

        Q[state_idx][block[1]] += 1/N[(block[0], block[1])]*(G[i] - Q[state_idx][block[1]]) 

    state_in_episode = set()
    for block in chain:
        if block[0] not in state_in_episode:
            state_in_episode.add(block[0])
            update_policy(block[0])

    episode += 1

print(results)

data = {'dealer_show':[], 'player_sum':[], 'v*':[], 'action':[]}
for i, q in enumerate(Q):
    data['dealer_show'].append(S[i][0])
    data['player_sum'].append(S[i][1])
    data['v*'].append(max(q))
    data['action'].append(1 if q[1]>q[0] else (0 if q[0]>q[1] else 2))
df = pd.DataFrame(data=data)
df.to_csv('out.csv', index=False)

