import matplotlib.pyplot as plt
import numpy as np

def reward_distance(minimumDistance):
    roadRadius = 36
    reward = abs(1 - (minimumDistance/roadRadius))
    alpha = 1
    return (-alpha*(reward**2))

x = [0]
N = 100
h = 72/100
x0 = x[0]
for i in range(N):
    x0 += h
    x.append(x0)  
   
y = []
for i in x:
    y.append(reward_distance(i))
    
print(y)
    
plt.figure(figsize=(5, 3), dpi=120)
plt.plot(x, y)
plt.ylabel("Distance reward")
plt.axvline(x=36, color = 'r', label = 'Center of the Track')
plt.grid(True)

plt.xlabel("Position in the cross section (in pixels)")
plt.ylabel("Reward Value")
#plt.title("Distance reward over the cross section of the track")
plt.legend()
plt.show()
