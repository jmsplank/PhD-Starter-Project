import numpy as np
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[0.6, 0.20], [0.2, 0.2]]

x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x, y, "x", color="r", alpha=0.2)

data = np.transpose(np.array([x, y]))

M = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        M[i, j] = np.mean(data[:, i] * data[:, j]) - np.mean(data[:, i]) * np.mean(
            data[:, j]
        )

w, v = np.linalg.eig(M)

data_trans = np.transpose(np.dot(v.T, data.T))
print(np.shape(data_trans))
plt.scatter(data_trans[:, 0], data_trans[:, 1], alpha=0.2)
# plt.show()

for e_, v_ in zip(w, v.T):
    plt.plot([0, 3 * e_ * v_[0]], [0, 3 * e_ * v_[1]], "r-", lw=2)

print(w)
print(v)


plt.axis("equal")
plt.show()
