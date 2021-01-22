import random

def draw_card():
    card = random.randint(1, 10)*(1 if random.randint(1,3) < 3 else -1)
    return card

def step(s, a):
    '''
    s: current state tuple of dealer's first card and the player's sum (x, y)
    a: action (0 - stick, 1 - hit)
    '''
    r = 0
    is_terminated = False
    next_state = None

    if a == 1:
        next_state = (s[0], s[1] + draw_card())
        # check if user terminated
        if next_state[1] > 21 or next_state[1] < 1:
            r = -1
            is_terminated = True
    else:
        dealer_sum = s[0]
        while dealer_sum < 17 and dealer_sum >= 1:
            dealer_sum += draw_card()
        if dealer_sum < 1 or dealer_sum > 21 or dealer_sum < s[1]:
            r = 1
        elif s[1] == dealer_sum:
            r = 0
        else:
            r = -1

        is_terminated = True

    return is_terminated, next_state, r

def start_game():
    return (random.randint(1, 10), random.randint(1, 10))
