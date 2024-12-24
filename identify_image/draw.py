import numpy as np
import matplotlib.pyplot as plt

x=np.load("X_3_32_4.npy")
y1=np.load("valid_accc_3_32_2.npy")
y2=np.load("valid_accc_3_16.npy")
plt.figure(figsize=(60, 10), dpi=100)
plt.plot(x, y1, c='red', label="bc=16")
plt.plot(x, y2, c='green', linestyle='--', label="bc=32")
plt.legend(loc='best')
plt.scatter(x, y1, c='red')
plt.scatter(x, y2, c='green')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("epoch", fontdict={'size': 16})
plt.ylabel("acc", fontdict={'size': 16})
for a, b in zip(x, y1):
    plt.text(a, round(b,3), round(b,3), ha='center', va='bottom', fontsize=8, c='red')
for a, b in zip(x, y2):
    plt.text(a, round(b,3), round(b,3), ha='center', va='bottom', fontsize=8, c='green')
plt.show()