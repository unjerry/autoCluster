print("project autoCluster")
import torch
import numpy as np

np.set_printoptions(linewidth=np.inf)

print(torch.__version__)
print(torch.cuda.is_available())

import matplotlib.pyplot as plt


def normDist(x, p):
    return torch.norm((x - p), dim=-1)


# autoCluster model setup
import torch
import torch.nn as nn
import torch.nn.functional as F


class autoCluster(nn.Module):
    def __init__(self, x: torch.Tensor, k=2):
        super().__init__()
        self.k = k
        self.pointWeight = nn.Parameter(torch.randn([x.shape[0], k]))
        self.FUN = nn.Softmax(dim=-1)
        self.pointPose = nn.Parameter(
            torch.stack([torch.mean(x, dim=0) for _ in range(k)])
        )
        # print(self.pointPose, self.pointWeight)

    def forward(self, x):
        D = torch.stack(
            [normDist(x, self.pointPose[i, :]) ** 2 for i in range(self.k)],
            dim=1,
        )
        W = self.FUN(self.pointWeight)
        # W = (W - 1) ** 3 + 1
        # Wei = torch.sum(self.FUN(self.pointWeight), dim=0)
        imd = torch.einsum("ij,ij->ij", D, W)
        sum = torch.sum(imd, dim=0)
        # print(sum, imd)
        return sum


P = torch.tensor(
    [
        [-0.5, -0.5],
        [0.5, 0.5],
    ],
)
dPList = [
    torch.randn([20, 2]) * 0.1 + P[0, :],
    torch.randn([200, 2]) * 0.1 + P[1, :],
]
dP = torch.concatenate(dPList)
print(dP.shape)

# print(normDist(dP, P[0, :]))
# print(len(P.shape))

# start Train
import torch.optim as optim


# ------------
import sklearn
import numpy as np

np.random.seed(0)

import pandas as pd
import matplotlib.pyplot as plt

# # Loading the Dataset
#
# We are using the image dataset MNIST, which comprises images of handwritten digits. We use sklearn to fetch the dataset from openml.

from sklearn.datasets import fetch_openml

X, y = fetch_openml("mnist_784", return_X_y=True)

X

# Convert data to image pixel-by-pixel representation
X_images = X.to_numpy().reshape(X.shape[0], 28, 28)

# Flatten the data so that we can apply clustering
X = X_images.reshape(X.shape[0], -1)

# # Showing the Images (Digits)
#
# In the following, we display each of the digits in the data. First, we have to convert the data for the visualization. So, we transform the dataset back to its original shape of 70000x28x28, where each image has 28x28 pixels.
#
# However, note that to apply clustering, we have to use the "flattened" format with 784 features because we cannot apply clustering to datasets with more than two dimensions.

y = y.astype(int).to_numpy()

print(X.shape)

dP = torch.tensor(X) / 255.0
print(dP.shape)


K = 10
dP = dP.to("cuda")
KMEAN = autoCluster(dP, K)
KMEAN.to("cuda")
print(dP)
for _ in KMEAN.parameters():
    print(_)
# print(KMEAN(dP))
optimizer = optim.Adam(KMEAN.parameters(), lr=0.001)
for _ in range(100000):
    optimizer.zero_grad()
    ans = torch.sum(KMEAN(dP))
    ans.backward()
    optimizer.step()
    print(ans)

for _ in KMEAN.parameters():
    print(_)

P = KMEAN.pointPose.to("cpu").detach().numpy()
L = KMEAN(dP).to("cpu").detach().numpy()
Wei = torch.sum(KMEAN.FUN(KMEAN.pointWeight), dim=0).to("cpu").detach().numpy()
dP = dP.to("cpu")
fig, ax = plt.subplots(dpi=300)
ax.set_aspect(1.0)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
# ax.scatter(P[:, 0], P[:, 1])
ax.scatter(dP[:, 0], dP[:, 1])
fig.savefig("fig/test22.png")
print(KMEAN.FUN(KMEAN.pointWeight), L, Wei, sep="\n")
for _ in range(K):
    ax.add_artist(plt.Circle(P[_, :], L[_] / Wei[_], fill=False))
ax.scatter(P[:, 0], P[:, 1], s=0.5)
fig.savefig("fig/test21.png")

fig, ax = plt.subplots(dpi=300)
ax.set_aspect(1.0)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
# ax.scatter(P[:, 0], P[:, 1])
ax.scatter(dP[:, 2], dP[:, 3])
fig.savefig("fig/mnist/test23.png")
print(KMEAN.FUN(KMEAN.pointWeight), L, Wei, sep="\n")
for _ in range(K):
    ax.add_artist(plt.Circle(P[_, :], L[_] / Wei[_], fill=False))
ax.scatter(P[:, 2], P[:, 3], s=0.5)
fig.savefig("fig/mnist/test24.png")
