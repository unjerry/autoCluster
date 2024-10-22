import pickle
import os
import getopt
import sys
import json
from datetime import datetime

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
import torch.nn.functional as F

np.random.seed(0)
UniDevice = "cuda"


class test_field:
    def __init__(self):
        self.dataSet: torch.Tensor = None
        self.lableSet: torch.Tensor = None
        self.f = None
        self.f_inv = None
        self.N = None
        self.fOptimizer: torch.optim.Adam = None
        self.f_invOptimizer: torch.optim.Adam = None
        self.NOptimizer: torch.optim.Adam = None
        self.epoch = 0

    def gene_data_STPR_get(self) -> None:
        name = input("name:")

        with open("data/{:s}.json".format(name), "r") as file:
            data = json.load(file)

        print(data["column"])
        item_list = data["item"]

        for i in range(10):
            # print(item_list[i])
            print(datetime.fromtimestamp(int(item_list[i][0] / 1000)))

        time_X = [item[0] / 1000 for item in item_list]
        value_Y = [item[5] for item in item_list]

        print(time_X[:10])
        print(value_Y[:10])

        fig = plt.figure(figsize=(100, 5), dpi=300)
        plt.scatter(time_X, value_Y)
        plt.grid()
        fig.savefig("fig/close_of_{:s}.png".format(name))

        time_X = torch.tensor(time_X)
        value_Y = torch.tensor(value_Y)
        print(time_X.shape, value_Y.shape)
        DATA = torch.stack([time_X, value_Y], dim=1)
        print(DATA.shape)
        self.dataSet = DATA
        pass

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
        print("centers START", centers, radiuss, self.dataSet, "END", sep="\n")

    def gene_data_sine_uniform(self, k: int = 1, num: int = 100, SHf: int = 0) -> None:
        print("generating sine data")
        dataList: list[torch.Tensor] = []
        for _ in range(k):
            X = torch.rand(num * (_ + 1) ** 2) * 10
            # X = torch.arange(0, 10, 0.01)
            dataList.append(torch.stack([X, torch.sin(X) + 2 * _ + SHf], dim=1))
        self.dataSet = torch.concatenate(dataList)
        print("generating done")

    def gene_data_exp_uniform(self, k: int = 1, num: int = 100, SHf: int = 0) -> None:
        print("generating sine data")
        dataList: list[torch.Tensor] = []
        for _ in range(k):
            X = torch.rand(num * (_ + 1) ** 2) * 3
            # X = torch.arange(0, 10, 0.01)
            dataList.append(torch.stack([X, torch.exp(X) + 2 * _ + SHf], dim=1))
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
        print(f"draw data {name}")
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

    def draw_3D(
        self,
        data: torch.Tensor,
        name: str,
        lable_X: str = "X",
        lable_Y: str = "Y",
        lable_Z: str = "Z",
    ) -> None:
        print(f"draw data 3d {name}")
        plt.style.use("ggplot")
        ax = plt.subplot(111, projection="3d")
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        ax.set_zlabel(lable_Z)  # 坐标轴
        ax.set_ylabel(lable_Y)
        ax.set_xlabel(lable_X)
        plt.savefig(f"fig/draw_3D_{name}")
        plt.show()
        plt.close()
        print("draw done")
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

    class NormDot(nn.Module):
        "define a dot with a normal vector"

        def __init__(self, N_INPUT, N_OUTPUT):
            super().__init__()
            self.N = nn.Parameter(F.normalize(torch.rand([N_OUTPUT, N_INPUT])))

        def forward(self, x):
            with torch.no_grad():
                self.N.data = F.normalize(self.N)
            return torch.einsum("...j,ij->...i", x, self.N)

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

    def deTask(self, K: int) -> None:
        Xi: torch.Tensor = self.dataSet.to(UniDevice)  # (s,t)i
        X1i: torch.Tensor
        X2i: torch.Tensor
        ni: int = Xi.shape[0]
        print(f"[ni]={ni}")
        Ti: torch.Tensor = self.dataSet[:, 0:1].to(UniDevice)  # T
        STPRi: torch.Tensor = self.dataSet[:, 1:2].to(UniDevice)  # STPR for srock_price
        dSTPRdTi: torch.Tensor = torch.zeros([ni, 1], dtype=torch.float32).to(
            UniDevice
        )  # dSTPRdT for dSTPR/dT
        d2STPRdT2i: torch.Tensor = torch.zeros([ni, 1], dtype=torch.float32).to(
            UniDevice
        )  # for d2STPR/dT2

        pca = PCA(n_components=1)

        kdt = KDTree(Xi.to("cpu"), leaf_size=5, metric="euclidean")
        indices = kdt.query(Xi.to("cpu"), k=K + 1, return_distance=False)
        for i in range(ni):
            pca.fit(torch.index_select(Xi.to("cpu"), 0, torch.tensor(indices[i])))
            dSTPRdTi[i] = torch.tensor(pca.components_[0][1] / pca.components_[0][0])
        X1i = torch.concatenate([Ti, dSTPRdTi], dim=1)
        print(f"X1 shape {X1i.shape}")
        kdt = KDTree(X1i.to("cpu"), leaf_size=5, metric="euclidean")
        indices = kdt.query(X1i.to("cpu"), k=K + 1, return_distance=False)
        for i in range(ni):
            pca.fit(torch.index_select(X1i.to("cpu"), 0, torch.tensor(indices[i])))
            d2STPRdT2i[i] = torch.tensor(pca.components_[0][1] / pca.components_[0][0])
        X2i = torch.concatenate([Ti, d2STPRdT2i], dim=1)

        objc.draw_2D(Xi.to("cpu"), "_SineData")
        objc.draw_2D(X1i.to("cpu"), "_dSineData")
        objc.draw_2D(X2i.to("cpu"), "_d2SineData")

        FEAT = torch.concatenate([STPRi, d2STPRdT2i], dim=1)
        print(f"FEAT dim {FEAT.shape}")
        ax = plt.subplot(111, projection="3d")
        ax.scatter(
            STPRi.to("cpu").detach().numpy(),
            dSTPRdTi.to("cpu").detach().numpy(),
            d2STPRdT2i.to("cpu").detach().numpy(),
        )
        ax.set_zlabel("Z")  # 坐标轴
        ax.set_ylabel("Y")
        ax.set_xlabel("X")
        # plt.show()
        plt.savefig(f"fig/draw_3d")
        plt.close()
        pca = PCA(n_components=3)
        pca.fit(torch.concatenate([STPRi, d2STPRdT2i], dim=1).to("cpu"))
        print(pca.components_, pca.explained_variance_, pca.singular_values_)

        print(f"start epoch {self.epoch}")
        criterion = nn.MSELoss()
        L = None
        ONEE = torch.ones([FEAT.shape[0], 1], dtype=torch.float32).to(UniDevice)
        ZERO = torch.zeros([FEAT.shape[0], 1], dtype=torch.float32).to(UniDevice)
        for _ in range(10000):
            self.fOptimizer.zero_grad()

            RAND = torch.rand(FEAT.shape).to(UniDevice)
            FRAND = self.f(RAND)
            Fi = self.f(FEAT)
            L = criterion(Fi, ZERO) + 0.1 * criterion(FRAND, ONEE)
            L.backward()

            self.fOptimizer.step()
            pass
        print(f"end epoch with: L {L}")
        pass

    def deLinearTask(self, K: int) -> None:
        Xi: torch.Tensor = self.dataSet.to(UniDevice)  # (s,t)i
        X1i: torch.Tensor
        X2i: torch.Tensor
        ni: int = Xi.shape[0]
        print(f"[ni]={ni}")
        Ti: torch.Tensor = self.dataSet[:, 0:1].to(UniDevice)  # T
        STPRi: torch.Tensor = self.dataSet[:, 1:2].to(UniDevice)  # STPR for srock_price
        dSTPRdTi: torch.Tensor = torch.zeros([ni, 1], dtype=torch.float32).to(
            UniDevice
        )  # dSTPRdT for dSTPR/dT
        d2STPRdT2i: torch.Tensor = torch.zeros([ni, 1], dtype=torch.float32).to(
            UniDevice
        )  # for d2STPR/dT2

        pca = PCA(n_components=1)

        kdt = KDTree(Xi.to("cpu"), leaf_size=5, metric="euclidean")
        indices = kdt.query(Xi.to("cpu"), k=K + 1, return_distance=False)
        for i in range(ni):
            pca.fit(torch.index_select(Xi.to("cpu"), 0, torch.tensor(indices[i])))
            dSTPRdTi[i] = torch.tensor(pca.components_[0][1] / pca.components_[0][0])
        X1i = torch.concatenate([Ti, dSTPRdTi], dim=1)
        print(f"X1 shape {X1i.shape}")
        kdt = KDTree(X1i.to("cpu"), leaf_size=5, metric="euclidean")
        indices = kdt.query(X1i.to("cpu"), k=K + 1, return_distance=False)
        for i in range(ni):
            pca.fit(torch.index_select(X1i.to("cpu"), 0, torch.tensor(indices[i])))
            d2STPRdT2i[i] = torch.tensor(pca.components_[0][1] / pca.components_[0][0])
        X2i = torch.concatenate([Ti, d2STPRdT2i], dim=1)

        objc.draw_2D(Xi.to("cpu"), "_d0Data")
        objc.draw_2D(X1i.to("cpu"), "_d1Data")
        objc.draw_2D(X2i.to("cpu"), "_d2Data")

        FEAT = torch.concatenate([STPRi, dSTPRdTi, d2STPRdT2i], dim=1)
        print(f"FEAT dim {FEAT.shape}")
        ax = plt.subplot(111, projection="3d")
        ax.scatter(
            STPRi.to("cpu").detach().numpy(),
            dSTPRdTi.to("cpu").detach().numpy(),
            d2STPRdT2i.to("cpu").detach().numpy(),
        )
        ax.set_zlabel("Z")  # 坐标轴
        ax.set_ylabel("Y")
        ax.set_xlabel("X")
        plt.savefig(f"fig/draw_3d")
        plt.show()
        plt.close()
        pca = PCA(n_components=3)
        pca.fit(torch.concatenate([STPRi, dSTPRdTi, d2STPRdT2i], dim=1).to("cpu"))
        print(
            pca.components_,
            pca.explained_variance_ / pca.explained_variance_[0],
            pca.singular_values_ / pca.singular_values_[0],
            pca.explained_variance_ratio_ / pca.explained_variance_ratio_[0],
        )
        RATIO = pca.singular_values_ / pca.singular_values_[0]
        KL = 0
        while True or KL < 3:
            if RATIO[KL] < 0.5:
                break
            KL += 1
        print(KL)
        return torch.tensor(pca.components_[KL:], dtype=torch.float32)

    def deLinearAugTask(self) -> None:
        Pis: torch.Tensor = self.dataSet.to(UniDevice)  # (s,t)_i
        S: int = Pis.shape[1]
        Ti: torch.Tensor = self.f(Pis)
        Vis: torch.Tensor = self.f_inv(Ti)

        # train a normal autoEncoder
        criterion = nn.MSELoss()
        for _ in range(1000):
            self.fOptimizer.zero_grad()
            self.f_invOptimizer.zero_grad()

            Ti: torch.Tensor = self.f(Pis)
            Vis: torch.Tensor = self.f_inv(Ti)
            L: torch.Tensor = criterion(Vis, Pis)
            L.backward()
            print(f"iter:{_}, L {L}")

            self.fOptimizer.step()
            self.f_invOptimizer.step()

        # # visual the linearity
        Vis: torch.Tensor = self.f_inv(Ti)  # s<2
        VTis: torch.Tensor = vmap(jacrev(objc.f_inv))(Ti).view(-1, 2)
        VTTis: torch.Tensor = vmap(jacrev(jacrev(objc.f_inv)))(Ti).view(-1, 2)
        for _ in range(2):
            Vij: torch.Tensor = torch.stack(
                [Vis[:, _], VTis[:, _], VTTis[:, _]], dim=1
            )  # j<3
            self.draw_3D(
                Vij.to("cpu").detach().numpy(), f"_latentPlot{_}", "d0", "d1", "d2"
            )

        LAisj: torch.Tensor = torch.stack(
            [Vis, VTis, VTTis], dim=2
        )  # j<3 LA means latent

        self.N = self.NormDot(3, 1).to(UniDevice)
        self.NOptimizer = optim.Adam(self.N.parameters(), lr=0.0001)
        print("LATENT shape", LAisj.shape)
        print("LATENT dot shape", self.N(LAisj).shape)
        ZERO: torch.Tensor = torch.zeros_like(self.N(LAisj))
        for _ in range(1000):
            self.fOptimizer.zero_grad()
            self.f_invOptimizer.zero_grad()
            self.NOptimizer.zero_grad()

            Ti: torch.Tensor = self.f(Pis)
            Vis: torch.Tensor = self.f_inv(Ti)
            VTis: torch.Tensor = vmap(jacrev(objc.f_inv))(Ti).view(-1, 2)
            VTTis: torch.Tensor = vmap(jacrev(jacrev(objc.f_inv)))(Ti).view(-1, 2)
            LAisj: torch.Tensor = torch.stack(
                [Vis, VTis, VTTis], dim=2
            )  # j<3 LA means latent
            L: torch.Tensor = criterion(Vis, Pis) + criterion(self.N(LAisj), ZERO)
            L.backward()
            print(f"iter:{_}, L {L}")

            self.fOptimizer.step()
            self.f_invOptimizer.step()
            self.NOptimizer.step()

        # visual the linearity
        Vis: torch.Tensor = self.f_inv(Ti)  # s<2
        VTis: torch.Tensor = vmap(jacrev(objc.f_inv))(Ti).view(-1, 2)
        VTTis: torch.Tensor = vmap(jacrev(jacrev(objc.f_inv)))(Ti).view(-1, 2)
        for _ in range(S):
            Vij: torch.Tensor = torch.stack(
                [Vis[:, _], VTis[:, _], VTTis[:, _]], dim=1
            )  # j<3
            self.draw_3D(
                Vij.to("cpu").detach().numpy(), f"_latentPlot{_}", "d0", "d1", "d2"
            )
        for para in self.N.parameters():
            print("para", para, sep="\n")
        objc.draw_2D(Pis.to("cpu").detach().numpy(), "_circleDataInTask")
        objc.draw_2D(Vis.to("cpu").detach().numpy(), "_circleGeneInTask")

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
                objc.f = objc.FCN(2, 1, 16, 4)
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
            elif arg == "deFinder":
                objc.gene_data_sine_uniform(1, 500, 0)
                # objc.compute_pca(2)

                objc.f = objc.FCN(2, 1, 64, 4)  # the pde F
                objc.f_inv = objc.FCN(1, 1, 64, 4)  # the pde solution
                objc.fOptimizer = optim.Adam(objc.f.parameters(), lr=0.0001)
                objc.f_invOptimizer = optim.Adam(objc.f_inv.parameters(), lr=0.0001)

                objc.draw_2D(objc.dataSet[:, 0:2], "_SineData")

                objc.dataSet = torch.tensor(objc.dataSet, dtype=torch.float32)
                with open(datumst_file_path, "wb") as pkl_file:
                    pickle.dump(objc, pkl_file)
            elif arg == "deTrain":
                print("START deTraining")
                objc.f.to(UniDevice)
                while True:
                    objc.epoch += 1
                    objc.deTask(10)
                    with open(datumst_file_path, "wb") as pkl_file:
                        pickle.dump(objc, pkl_file)
                    print("save done", objc.epoch)
                    pass
            elif arg == "deGenerate":
                print("START generate")
                Ti = torch.arange(0, 5, 0.1).to(UniDevice).view(-1, 1)
                objc.f_inv.to(UniDevice)
                objc.f.to(UniDevice)
                criterion = nn.MSELoss()
                L = None
                ZERO = torch.zeros_like(Ti).to(UniDevice)
                s0 = torch.zeros([1]).to(UniDevice)
                ds0 = torch.ones([1]).to(UniDevice) * 0.5
                _ = 0
                while True:
                    _ += 1
                    objc.f_invOptimizer.zero_grad()

                    si = objc.f_inv(Ti)
                    dsi = vmap(jacrev(objc.f_inv))(Ti).view(-1, 1)
                    d2si = vmap(jacrev(jacrev(objc.f_inv)))(Ti).view(-1, 1)
                    # print(si, dsi, d2si, sep="\n")
                    FEAT = torch.concatenate([si, d2si], dim=1)
                    # print(FEAT.shape)
                    L = (
                        criterion(objc.f(FEAT), ZERO)
                        + 10 * criterion(si[0], s0)
                        + 10 * criterion(dsi[0], ds0)
                    )
                    L.backward()
                    print(f"iter:{_} L {L}")
                    if L < 1e-6:
                        break
                    if _ % 1000 == 0:
                        print(f"L {L*Ti.shape[0]}")
                        X1i = torch.concatenate([Ti, si], dim=1)
                        print(f"X1 shape {X1i.shape}")
                        objc.draw_2D(X1i.to("cpu").detach().numpy(), "_GeneredData")
                        X1i = torch.concatenate([Ti, objc.f(FEAT)], dim=1)
                        objc.draw_2D(X1i.to("cpu").detach().numpy(), "_GeneredEqu")
                        with open(datumst_file_path, "wb") as pkl_file:
                            pickle.dump(objc, pkl_file)

                    objc.f_invOptimizer.step()
                    pass
            elif arg == "deLinearSetupSine":
                print("SETUP deLinearTasks")
                objc.gene_data_sine_uniform(1, 100, 0)
                # objc.compute_pca(2)

                objc.f_inv = objc.FCN(1, 1, 64, 4)  # the pde solution
                objc.f_invOptimizer = optim.Adam(objc.f_inv.parameters(), lr=0.0001)

                objc.draw_2D(objc.dataSet[:, 0:2], "_SineData")

                objc.dataSet = torch.tensor(objc.dataSet, dtype=torch.float32)
                with open(datumst_file_path, "wb") as pkl_file:
                    pickle.dump(objc, pkl_file)
            elif arg == "deLinearSetupExp":
                print("SETUP deLinearTasks")
                objc.gene_data_exp_uniform(1, 500, 0)
                # objc.compute_pca(2)

                objc.f_inv = objc.FCN(1, 1, 64, 4)  # the pde solution
                objc.f_invOptimizer = optim.Adam(objc.f_inv.parameters(), lr=0.0001)

                objc.draw_2D(objc.dataSet[:, 0:2], "_ExpData")

                objc.dataSet = torch.tensor(objc.dataSet, dtype=torch.float32)
                with open(datumst_file_path, "wb") as pkl_file:
                    pickle.dump(objc, pkl_file)
            elif arg == "deLinearSetupSTPR":
                print("SETUP deLinearTasks")
                objc.gene_data_STPR_get()
                # objc.compute_pca(2)

                objc.f_inv = objc.FCN(1, 1, 64, 4)  # the pde solution
                objc.f_invOptimizer = optim.Adam(objc.f_inv.parameters(), lr=0.0001)

                objc.draw_2D(objc.dataSet[:, 0:2], "_STPRData")

                objc.dataSet = torch.tensor(objc.dataSet, dtype=torch.float32)
                with open(datumst_file_path, "wb") as pkl_file:
                    pickle.dump(objc, pkl_file)
            elif arg == "deLinearTrain":
                print("START deLinearTraining")
                objc.f = objc.deLinearTask(10).to(UniDevice)
                with open(datumst_file_path, "wb") as pkl_file:
                    pickle.dump(objc, pkl_file)
            elif arg == "deLinearGenerate":
                print("START generate")
                Ti = torch.arange(0, 5, 0.1).to(UniDevice).view(-1, 1)
                objc.f_inv.to(UniDevice)
                objc.f = objc.f.to(UniDevice)
                print(objc.f)
                criterion = nn.MSELoss()
                L = None
                ZERO = torch.zeros_like(Ti).to(UniDevice)
                s0 = torch.ones([1]).to(UniDevice) * 0.5
                ds0 = torch.ones([1]).to(UniDevice) * 0.5
                _ = 0
                while True:
                    _ += 1
                    objc.f_invOptimizer.zero_grad()

                    si = objc.f_inv(Ti)
                    dsi = vmap(jacrev(objc.f_inv))(Ti).view(-1, 1)
                    d2si = vmap(jacrev(jacrev(objc.f_inv)))(Ti).view(-1, 1)
                    # print(si, dsi, d2si, sep="\n")
                    FEAT = torch.concatenate([si, dsi, d2si], dim=1)
                    # print(FEAT.shape)
                    L = (
                        criterion(
                            torch.einsum("ji,ki->j", FEAT, objc.f).view(-1, 1), ZERO
                        )
                        + 10 * criterion(si[0], s0)
                        + 1000 * criterion(dsi[0], ds0)
                    )
                    L.backward()
                    print(f"iter:{_} L {L}")
                    if L < 1e-6:
                        break
                    if _ % 1000 == 0:
                        print(f"L {L*Ti.shape[0]}")
                        X1i = torch.concatenate([Ti, si], dim=1)
                        print(f"X1 shape {X1i.shape}")
                        objc.draw_2D(X1i.to("cpu").detach().numpy(), "_GeneredData")
                        X1i = torch.concatenate(
                            [Ti, torch.einsum("ji,ki->j", FEAT, objc.f).view(-1, 1)],
                            dim=1,
                        )
                        objc.draw_2D(X1i.to("cpu").detach().numpy(), "_GeneredEqu")
                        with open(datumst_file_path, "wb") as pkl_file:
                            pickle.dump(objc, pkl_file)

                    objc.f_invOptimizer.step()

                print(f"L {L*Ti.shape[0]}")
                X1i = torch.concatenate([Ti, si], dim=1)
                print(f"X1 shape {X1i.shape}")
                objc.draw_2D(X1i.to("cpu").detach().numpy(), "_GeneredData")
                X1i = torch.concatenate(
                    [Ti, torch.einsum("ji,ki->j", FEAT, objc.f).view(-1, 1)], dim=1
                )
                objc.draw_2D(X1i.to("cpu").detach().numpy(), "_GeneredEqu")
                with open(datumst_file_path, "wb") as pkl_file:
                    pickle.dump(objc, pkl_file)
            elif arg == "deLinearAugSetupCircle":
                print("SETUP deLinearTasks")
                objc.gene_data_circle_uniform(2, 1, 100)
                objc.compute_pca(2)

                objc.f_inv = objc.FCN(1, 2, 128, 4)  # the parametrize decoder
                objc.f = objc.FCN(2, 1, 128, 4)
                objc.f_invOptimizer = optim.Adam(objc.f_inv.parameters(), lr=0.0001)
                objc.fOptimizer = optim.Adam(objc.f.parameters(), lr=0.0001)

                objc.draw_2D(objc.dataSet[:, 0:2], "_circleData")

                objc.dataSet = torch.tensor(objc.dataSet, dtype=torch.float32)

                with open(datumst_file_path, "wb") as pkl_file:
                    pickle.dump(objc, pkl_file)
            elif arg == "deLinearAugTrain":
                print("START deLinearAugTrain")
                objc.f.to(UniDevice)
                objc.f_inv.to(UniDevice)
                objc.deLinearAugTask()
                with open(datumst_file_path, "wb") as pkl_file:
                    pickle.dump(objc, pkl_file)

    for value in values:
        if value == "regenerate":
            print("regenerating")
            objc = test_field()
            with open(datumst_file_path, "wb") as pkl_file:
                pickle.dump(objc, pkl_file)
            print("done")
