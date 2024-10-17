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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
dP = iris["data"]
y = iris["target"]
names = iris["target_names"]
feature_names = iris["feature_names"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(dP)
# ------------
dP = torch.tensor(dP[:, :])
print(dP.shape)


K = 3
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
fig.savefig("fig/test23.png")
print(KMEAN.FUN(KMEAN.pointWeight), L, Wei, sep="\n")
for _ in range(K):
    ax.add_artist(plt.Circle(P[_, :], L[_] / Wei[_], fill=False))
ax.scatter(P[:, 2], P[:, 3], s=0.5)
fig.savefig("fig/test24.png")
