# Reference:https://janakiev.com/blog/pytorch-iris/


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")


# ----------------------


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris["data"]
y = iris["target"]
names = iris["target_names"]
feature_names = iris["feature_names"]

print(X, y)

# Scale data to have mean 0 and variance 1
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2
)


# --------------------


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for target, target_name in enumerate(names):
    print(target,y == target)
    X_plot = X[y == target]
    ax1.plot(
        X_plot[:, 0], X_plot[:, 1], linestyle="none", marker="o", label=target_name
    )
ax1.set_xlabel(feature_names[0])
ax1.set_ylabel(feature_names[1])
ax1.axis("equal")
ax1.legend()

for target, target_name in enumerate(names):
    X_plot = X[y == target]
    print(len(X_plot))
    ax2.plot(
        X_plot[:, 2], X_plot[:, 3], linestyle="none", marker="o", label=target_name
    )
ax2.set_xlabel(feature_names[2])
ax2.set_ylabel(feature_names[3])
ax2.axis("equal")
ax2.legend()

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for target, target_name in enumerate(names):
    print(target,y == target)
    X_plot = X[y == target]
    ax1.plot(
        X_plot[:, 0], X_plot[:, 2], linestyle="none", marker="o", label=target_name
    )
ax1.set_xlabel(feature_names[0])
ax1.set_ylabel(feature_names[1])
ax1.axis("equal")
ax1.legend()

for target, target_name in enumerate(names):
    X_plot = X[y == target]
    print(len(X_plot))
    ax2.plot(
        X_plot[:, 1], X_plot[:, 3], linestyle="none", marker="o", label=target_name
    )
ax2.set_xlabel(feature_names[2])
ax2.set_ylabel(feature_names[3])
ax2.axis("equal")
ax2.legend()

plt.show()
