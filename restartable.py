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

np.random.seed(0)


class test_field:
    def __init__(self):
        self.dataSet: torch.Tensor
        self.lableSet: torch.Tensor

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
        self.dataSet = X
        self.lableSet = y
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

    objc.get_data_MNIST()
    objc.draw_2D_MNIST(objc.dataSet[:, 0:2], objc.lableSet, "[0,1]")

    objc.compute_pca(2)
    objc.draw_2D_MNIST(objc.dataSet[:, 0:2], objc.lableSet, "[pca]")

    # sd

    print("running")
    print("sys.argv", sys.argv)
    short_options = "ho:v"
    long_options = ["help", "output=", "verbose"]
    try:
        # 解析命令行参数
        args, values = getopt.getopt(sys.argv[1:], "ho:", ["help", "output="])
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

    for value in values:
        if value == "regenerate":
            print("regenerating")
            objc = test_field()
            with open(datumst_file_path, "wb") as pkl_file:
                pickle.dump(objc, pkl_file)
            print("done")
