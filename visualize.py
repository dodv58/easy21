import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

print(sys.argv)
if len(sys.argv) < 2:
    print('Missing arguments')
elif sys.argv[1] == '1':
    df = pd.read_csv('out.csv')
    sns.set(style = "darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    x = df['dealer_show']
    y = df['player_sum']
    z = df['v*']

    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_zlabel("v*")

    ax.scatter(x, y, z)

    plt.show()
else:
    df = pd.read_csv('out.csv')
    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure()

    sns.swarmplot(data=df, x="dealer_show", y="player_sum", hue="action")

    plt.show()
