import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

'''

sns.set_theme(style="whitegrid", palette="muted")
fig = plt.figure()

sns.swarmplot(data=df, x="dealer_show", y="player_sum", hue="action")

plt.show()
'''
