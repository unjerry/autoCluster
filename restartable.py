import pickle
import os
import getopt
import sys

import torch
import matplotlib.pyplot as plt

import sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from torch.func import jacrev, vmap
import torch.nn as nn
import torch.optim as optim

np.random.seed(0)
UniDevice = "cuda"


class test_field:
    def __init__(self):
        self.dataSet: torch.Tensor = None
        self.lableSet: torch.Tensor = None
        self.f = None
        self.f_inv = None
        self.fOptimizer = None
        self.f_invOptimizer = None

    def gene_data_circle_uniform(
        self, dimension: int = 2, k: int = 3, each_N: int = 100
    ) -> None:
        centers: torch.Tensor = torch.rand([k, dimension])
        radiuss: torch.Tensor = torch.rand([k])
        lis: list[torch.Tensor] = []
        for i in range(k):
            theta = torch.rand([each_N]) * 2 * torch.pi
            z = torch.exp(1j * theta) * radiuss[i] + (
                centers[i][0] + centers[i][1] * 1j
            )
            lis.append(torch.stack([z.real, z.imag], dim=1))
        self.dataSet = torch.concatenate(lis, dim=0)
        print("centers", centers, radiuss, self.dataSet, sep="\n")

    def gene_data_sine_uniform(self, k: int = 1, num: int = 100, SHf: int = 0) -> None:
        print("generating sine data")
        dataList: list[torch.Tensor] = []
        for _ in range(k):
            X = torch.rand(num * (_ + 1) ** 2) * 10
            dataList.append(torch.stack([X, torch.sin(X) + 2 * _ + SHf], dim=1))
        self.dataSet = torch.concatenate(dataList)
        print("generating done")

    def draw_data_circle(self) -> None:
        print("draw data")
        plt.style.use("ggplot")
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        ax1.plot(
            self.dataSet[:, 0],
            self.dataSet[:, 1],
            linestyle="none",
            marker="o",
            label="Y",
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.axis("equal")
        ax1.legend()
        plt.savefig("fig/circle/drawData.png")
        print("draw done")
        plt.close()

    def get_data_MNIST(self) -> None:
        X, y = fetch_openml("mnist_784", return_X_y=True)
        # Convert data to image pixel-by-pixel representation
        X_images = X.to_numpy().reshape(X.shape[0], 28, 28)

        # Flatten the data so that we can apply clustering
        X = X_images.reshape(X.shape[0], -1)
        y = y.astype(int).to_numpy()
        print(X, y, sep="\n")
        print(X.shape, y.shape, sep="\n")
        self.dataSet = torch.tensor(X, dtype=torch.float32)
        self.lableSet = torch.tensor(y)
        pass

    def draw_2D(self, data: torch.Tensor, name: str) -> None:
        print("draw data")
        plt.style.use("ggplot")
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        ax1.plot(
            data[:, 0],
            data[:, 1],
            linestyle="none",
            marker="o",
            label="Y",
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.axis("equal")
        ax1.legend()
        plt.savefig(f"fig/draw_2D_{name}")
        print("draw done")
        plt.close()
        pass

    def draw_1D(self, data: torch.Tensor, name: str) -> None:
        print("draw data")
        plt.style.use("ggplot")
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        ax1.plot(
            data[:, 0],
            np.zeros_like(data[:, 0]),
            linestyle="none",
            marker="o",
            label="Y",
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.axis("equal")
        ax1.legend()
        plt.savefig(f"fig/draw_2D_{name}")
        print("draw done")
        plt.close()
        pass

    def draw_2D_MNIST(self, data: torch.Tensor, label: torch.Tensor, name: str) -> None:
        colors = [
            "blue",
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        markers = [
            "+",
            "-",
            "o",
            "s",
            "^",
            "*",
            "%",
            "$",
            "#",
            "@",
        ]
        print("draw data")
        plt.style.use("ggplot")
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        ax1.scatter(
            data[:, 0],
            data[:, 1],
            c=[colors[label[i]] for i in range(len(label))],
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.axis("equal")
        ax1.legend()
        plt.savefig(f"fig/draw_2D_MNIST_{name}")
        print("draw done")
        plt.close()
        pass

    def compute_pca(self, dim: int) -> None:
        pca = PCA(n_components=dim)
        pca.fit(self.dataSet)
        self.dataSet = pca.fit_transform(self.dataSet)
        pass

    class FCN(nn.Module):
        "define a fully connected network"

        def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
            super().__init__()
            activation = nn.Tanh
            self.fcs = nn.Sequential(*[nn.Linear(N_INPUT, N_HIDDEN), activation()])
            self.fch = nn.Sequential(
                *[
                    nn.Sequential(*[nn.Linear(N_HIDDEN, N_HIDDEN), activation()])
                    for _ in range(N_LAYERS)
                ]
            )
            self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

        def forward(self, x):
            x = self.fcs(x)
            x = self.fch(x)
            x = self.fce(x)
            return x

    def localDis(self, P: torch.Tensor, Q: torch.Tensor):
        return torch.dist(P, Q)

    def localPCAIndexDis(self, A: torch.Tensor, I: torch.Tensor, Q: torch.Tensor):
        P = torch.index_select(A, 0, I)
        # print("SHAPE", Q.shape, P.shape)
        # assert P.shape == (5, 1)
        # assert Q.shape == (1,)
        return vmap(self.localDis, in_dims=(0, None))(P, Q)

    def localPCAtask(self, K: int, S: int) -> None:
        pca = PCA(n_components=S)
        nj: int = self.dataSet.shape[0]
        ni: int = self.dataSet.shape[1]
        print(f"[nj,ni]={nj,ni}")
        Xji = self.dataSet.to(UniDevice)  # [j,i]<[N,2]
        Ejsi: torch.Tensor = torch.zeros(
            [nj, S, ni], device=UniDevice
        )  # [j,k,i]<[nj,S,ni]
        print("START NEAREST")
        nbrs = kdt = KDTree(Xji.to("cpu"), leaf_size=3, metric="euclidean")
        distances, indices = kdt.query(Xji.to("cpu"), k=K + 1, return_distance=True)
        # print("IND", indices, distances)
        # print("SHAPE", distances.shape)
        RHOjk = torch.tensor(distances, dtype=torch.float32).to(UniDevice)[:, 1:]
        Ijk = torch.tensor(indices).to(UniDevice)[:, 1:]
        # print("RHO", RHOjk)
        for j in range(nj):
            pca.fit(torch.index_select(self.dataSet, 0, torch.tensor(indices[j])))
            Ejsi[j] = torch.tensor(pca.components_)
        # print("Xji,Ejsi", Xji, Ejsi, Xji.shape, Ejsi.shape, sep="\n")
        Ejis = torch.einsum("jsi->jis", Ejsi)
        assert Ejis.shape == (nj, ni, S)

        print("localPCAtask draw")
        plt.style.use("ggplot")
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        ax1.quiver(
            self.dataSet[:, 0].to("cpu"),
            self.dataSet[:, 1].to("cpu"),
            Ejsi[:, 0, 0].to("cpu"),
            Ejsi[:, 0, 1].to("cpu"),
            angles="xy",
            scale_units="xy",
            minlength=1,
            scale=1,
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.axis("equal")
        plt.savefig(f"fig/localPCAtask_draw_quver")
        print("done draw")
        plt.close()

        x = self.f_inv(Xji)
        Jjis = vmap(jacrev(self.f))(x)
        assert Jjis.shape == (nj, ni, S)

        # train
        criterion = nn.MSELoss()
        L = None
        ONEE = torch.ones(nj, S).to(UniDevice)
        for _ in range(1000):
            self.fOptimizer.zero_grad()
            self.f_invOptimizer.zero_grad()

            tjs: torch.Tensor = self.f_inv(Xji)
            SIGMjk = vmap(self.localPCAIndexDis, in_dims=(None, 0, 0))(
                tjs,
                Ijk,
                tjs,
            )
            Jjis = vmap(jacrev(self.f))(tjs)
            Yji = self.f(tjs)
            assert SIGMjk.shape == (nj, K)
            assert RHOjk.shape == (nj, K)
            # with torch.no_grad():
            #     print(tjs[0:10, :], SIGMjk.to("cpu")[0:10, :], RHOjk.to("cpu")[0:10, :])
            L = (
                +1.0 * criterion(Xji, Yji)
                + 0.1
                * criterion(
                    torch.abs(torch.einsum("jis,jis->js", Ejis, Jjis)),
                    ONEE,
                )
                + 1.0 * criterion(SIGMjk, RHOjk)
            )
            L.backward()
            print("iter:", _, L)

            self.fOptimizer.step()
            self.f_invOptimizer.step()
            pass
        print("loss", L)
        pass


if __name__ == "__main__":
    # loading the data into objc
    print("loading")
    objc = None
    # get current file path and the folder path
    current_file_path = os.path.abspath(__file__)
    current_file_fold = os.path.split(current_file_path)[0]
    datumst_file_path = os.path.join(current_file_fold, *["restartable.datums"])
    print(f"finding data file:{datumst_file_path}")
    if os.path.exists(datumst_file_path):
        print("data exists, loading")
        with open(datumst_file_path, "rb") as pkl_file:
            objc = pickle.load(pkl_file)
        print("data loading done")
    else:
        print("data doesn't exist, creating")
        objc = test_field()
        with open(datumst_file_path, "wb") as pkl_file:
            pickle.dump(objc, pkl_file)
        print("creating done")
    print("load done")
    assert objc != None

    # sd

    # sd

    print("running")
    print("sys.argv", sys.argv)
    short_options = "ho:vt:"
    long_options = ["help", "output=", "verbose", "train="]
    try:
        # 解析命令行参数
        args, values = getopt.getopt(sys.argv[1:], short_options, long_options)
    except getopt.GetoptError as err:
        print("ERROR:", str(err))  # 打印错误信息
        sys.exit(2)

    for opt, arg in args:
        if opt in ("-h", "--help"):
            print("I'm helping you.")
        elif opt in ("-o", "--output"):
            print("output dir:", arg)
        elif opt == "-v":
            print("I don't know what is this.")
        elif opt in ("-t", "--train"):
            if arg == "localPCAtrain":
                epoch = 1
                objc.dataSet.to(UniDevice)
                objc.f.to(UniDevice)
                objc.f_inv.to(UniDevice)
                while True:
                    print("data_shape", objc.dataSet.shape)
                    objc.draw_2D(objc.dataSet[:, 0:2], "_[0,1]")
                    # for _ in objc.f.parameters():
                    #     print(_)
                    # input("press arbitary key to continue")
                    print("running", epoch)
                    objc.localPCAtask(5, 1)
                    # print(objc.f.parameters())
                    # for _ in objc.f.parameters():
                    #     print(_)

                    with open(datumst_file_path, "wb") as pkl_file:
                        pickle.dump(objc, pkl_file)
                    print("save done", epoch)

                    objc.draw_2D(objc.dataSet.to("cpu").detach().numpy(), f"xi{epoch}")
                    t = objc.f_inv(objc.dataSet.to(UniDevice))
                    print(torch.min(t).tolist(), torch.max(t).tolist())
                    objc.draw_2D(
                        objc.f(
                            torch.range(
                                torch.min(t).tolist() - 1,
                                torch.max(t).tolist() + 1,
                                0.001,
                            )
                            .to(UniDevice)
                            .view(-1, 1)
                        )
                        .to("cpu")
                        .detach()
                        .numpy(),
                        f"sm{epoch}",
                    )
                    objc.draw_1D(t.to("cpu").detach().numpy(), f"ts{epoch}")

                    y = objc.f(t)
                    objc.draw_2D(y.to("cpu").detach().numpy(), f"yi{epoch}")

                    x = objc.f_inv(objc.dataSet.to(UniDevice))
                    Jjis = vmap(jacrev(objc.f))(x)
                    # assert Jjis.shape == (nj, ni, S)
                    print("localPCAtask train draw")
                    plt.style.use("ggplot")
                    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
                    ax1.quiver(
                        objc.dataSet[:, 0].to("cpu").detach().numpy(),
                        objc.dataSet[:, 1].to("cpu").detach().numpy(),
                        Jjis[:, 0, :].to("cpu").detach().numpy(),
                        Jjis[:, 1, :].to("cpu").detach().numpy(),
                        angles="xy",
                        scale_units="xy",
                        minlength=1,
                        scale=1,
                    )
                    ax1.set_xlabel("X")
                    ax1.set_ylabel("Y")
                    ax1.axis("equal")
                    plt.savefig(f"fig/localPCAtask_draw_train_quver{epoch}")
                    print("done draw")
                    plt.close()

                    epoch += 1
            elif arg == "localPCASetup":
                objc.gene_data_sine_uniform(1, 100, 5)
                objc.f = objc.FCN(1, 2, 64, 2)
                objc.f_inv = objc.FCN(2, 1, 64, 2)
                objc.fOptimizer = optim.Adam(objc.f.parameters(), lr=0.0001)
                objc.f_invOptimizer = optim.Adam(objc.f_inv.parameters(), lr=0.0001)
                # objc.compute_pca(2)
                objc.dataSet = torch.tensor(objc.dataSet, dtype=torch.float32)
                with open(datumst_file_path, "wb") as pkl_file:
                    pickle.dump(objc, pkl_file)
            elif arg == "localPCAcircleSetup":
                objc.gene_data_circle_uniform(2, 1)
                objc.f = objc.FCN(1, 2, 64, 2)
                objc.f_inv = objc.FCN(2, 1, 64, 2)
                objc.fOptimizer = optim.Adam(objc.f.parameters(), lr=0.0001)
                objc.f_invOptimizer = optim.Adam(objc.f_inv.parameters(), lr=0.0001)
                objc.compute_pca(2)
                objc.dataSet = torch.tensor(objc.dataSet, dtype=torch.float32)
                with open(datumst_file_path, "wb") as pkl_file:
                    pickle.dump(objc, pkl_file)
            elif arg == "localPCAmnistSetup":
                objc.get_data_MNIST()
                objc.draw_2D_MNIST(objc.dataSet[:, 0:2], objc.lableSet, "[0,1]")
                # objc.compute_pca(2)
                # objc.draw_2D_MNIST(objc.dataSet[:, 0:2], objc.lableSet, "[pca]")
                objc.f = objc.FCN(10, 784, 64, 2)
                objc.f_inv = objc.FCN(784, 10, 64, 2)
                objc.fOptimizer = optim.Adam(objc.f.parameters(), lr=0.0001)
                objc.f_invOptimizer = optim.Adam(objc.f_inv.parameters(), lr=0.0001)
                objc.dataSet = torch.tensor(objc.dataSet[:, :], dtype=torch.float32)
                with open(datumst_file_path, "wb") as pkl_file:
                    pickle.dump(objc, pkl_file)

    for value in values:
        if value == "regenerate":
            print("regenerating")
            objc = test_field()
            with open(datumst_file_path, "wb") as pkl_file:
                pickle.dump(objc, pkl_file)
            print("done")
