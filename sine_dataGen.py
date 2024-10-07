import torch
import matplotlib

# matplotlib.use("AGG")
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim
import numpy as np

np.set_printoptions(linewidth=201)


class sineTest:
    def __init__(self) -> None:
        self.dataSet: torch.Tensor
        self.encoder: self.sineCoder = self.sineCoder(inDim=2, outDim=4)
        self.decoder: self.sineCoder = self.sineCoder(inDim=4, outDim=2)
        self.KMEAN: self.autoCluster
        self.parameter: torch.Tensor
        self.output: torch.Tensor

    def generateData(self, k: int = 3, num: int = 100) -> None:
        print("generating data")
        dataList: list[torch.Tensor] = []
        for _ in range(k):
            X = torch.rand(num) * 10
            dataList.append(torch.stack([X, torch.sin(X) + 1 * _ + 10], dim=1))
        self.dataSet = torch.concatenate(dataList)
        print("generating done")

    def drawData(self) -> None:
        print("draw data")
        plt.style.use("ggplot")
        fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
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
        plt.savefig("fig/sine/drawData.png")
        plt.show()
        print("draw done")
        plt.close()

    def drawOutput(self, num: int) -> None:
        print("draw output")
        plt.style.use("ggplot")
        fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
        ax1.plot(
            self.output.detach().numpy()[0 * 100 : (0 + 1) * 100, 0],
            self.output.detach().numpy()[0 * 100 : (0 + 1) * 100, 1],
            linestyle="none",
            marker="o",
            label="Y",
            c="yellow",
        )
        ax1.plot(
            self.output.detach().numpy()[1 * 100 : (1 + 1) * 100, 0],
            self.output.detach().numpy()[1 * 100 : (1 + 1) * 100, 1],
            linestyle="none",
            marker="o",
            label="Y",
            c="blue",
        )
        ax1.plot(
            self.output.detach().numpy()[2 * 100 : (2 + 1) * 100, 0],
            self.output.detach().numpy()[2 * 100 : (2 + 1) * 100, 1],
            linestyle="none",
            marker="o",
            label="Y",
            c="green",
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.axis("equal")
        ax1.legend()
        plt.savefig(f"fig/sine/drawOutput/{num}.png")
        # plt.show()
        print("draw done")
        plt.close()

    def drawParameter(self, num: int) -> None:
        col = ["yellow", "blue", "green"]
        print("draw parameter")
        plt.style.use("ggplot")
        fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
        ax1.plot(
            self.parameter.detach().numpy()[0 * 100 : (0 + 1) * 100, 0],
            self.parameter.detach().numpy()[0 * 100 : (0 + 1) * 100, 1],
            linestyle="none",
            marker="o",
            label="Y",
            c="yellow",
        )
        # ax1.plot(
        #     self.parameter.detach().numpy()[0 * 100 : (0 + 1) * 100, 0],
        #     np.ones_like(self.parameter.detach().numpy()[0 * 100 : (0 + 1) * 100, 0])
        #     * 1,
        #     linestyle="none",
        #     marker="o",
        #     label="Y",
        #     c="yellow",
        # )
        ax1.plot(
            self.parameter.detach().numpy()[1 * 100 : (1 + 1) * 100, 0],
            self.parameter.detach().numpy()[1 * 100 : (1 + 1) * 100, 1],
            linestyle="none",
            marker="o",
            label="Y",
            c="blue",
        )
        # ax1.plot(
        #     self.parameter.detach().numpy()[1 * 100 : (1 + 1) * 100, 0],
        #     np.ones_like(self.parameter.detach().numpy()[0 * 100 : (0 + 1) * 100, 0])
        #     * 2,
        #     linestyle="none",
        #     marker="o",
        #     label="Y",
        #     c="blue",
        # )
        ax1.plot(
            self.parameter.detach().numpy()[2 * 100 : (2 + 1) * 100, 0],
            self.parameter.detach().numpy()[2 * 100 : (2 + 1) * 100, 1],
            linestyle="none",
            marker="o",
            label="Y",
            c="green",
        )
        # ax1.plot(
        #     self.parameter.detach().numpy()[2 * 100 : (2 + 1) * 100, 0],
        #     np.ones_like(self.parameter.detach().numpy()[0 * 100 : (0 + 1) * 100, 0])
        #     * 3,
        #     linestyle="none",
        #     marker="o",
        #     label="Y",
        #     c="green",
        # )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        # ax1.axis("equal")
        ax1.legend()
        plt.savefig(f"fig/sine/drawParameter/{num}.png")
        # plt.show()
        print("draw done")
        plt.close()
        P = self.KMEAN.pointPose.to("cpu").detach().numpy()
        # L = self.KMEAN(self.parameter).to("cpu").detach().numpy()
        Wei = (
            torch.argmax(self.KMEAN.FUN(self.KMEAN.pointWeight), dim=1)
            .to("cpu")
            .detach()
            .numpy()
        )
        print(Wei)
        self.parameter = self.parameter.detach().numpy()
        fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
        # # ax.set_aspect(1.0)
        # ax1.axis("equal")
        # # ax.set_xlim([0, 10])
        # # ax.set_ylim([0, 10])
        print(P.shape)
        ax1.scatter(P[:, 0], P[:, 1])
        for i in range(self.parameter.shape[0]):
            ax1.scatter(self.parameter[i, 0], self.parameter[i, 1], c=col[Wei[i]])
        # ax1.axis("equal")
        fig.savefig(f"fig/sine/P/{num}.png")
        # print(self.KMEAN.FUN(self.KMEAN.pointWeight), L, Wei, sep="\n")
        # for _ in range(3):
        #     ax.add_artist(plt.Circle(P[_], L[_] / Wei[_], fill=False, color="black"))
        # ax.scatter(P[:, 0], P[:, 0])
        # ax.axis("equal")
        # ax.legend()
        # fig.savefig("fig/sine/test24.png")

    class sineCoder(nn.Module):
        def __init__(
            self, inDim: int = 2, outDim: int = 2, hidDim=16, hidNum=3, *args, **kwargs
        ) -> None:
            super().__init__(*args, **kwargs)
            # self.scale = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            self.ipl = nn.Sequential(
                OrderedDict(
                    [
                        ("inputLinear", nn.Linear(inDim, hidDim)),
                        ("inputActivation", nn.Sigmoid()),
                    ]
                )
            )
            self.hpl = nn.Sequential(
                OrderedDict(
                    [
                        (
                            f"hidden{i}",
                            nn.Sequential(
                                OrderedDict(
                                    [
                                        ("Linear", nn.Linear(hidDim, hidDim)),
                                        ("Activation", nn.Softplus()),
                                    ]
                                )
                            ),
                        )
                        for i in range(hidNum)
                    ]
                )
            )
            self.opl = nn.Sequential(
                OrderedDict(
                    [
                        ("outputLinear", nn.Linear(hidDim, outDim)),
                        ("outputActivation", nn.Softplus()),
                    ]
                )
            )

        def forward(self, x):
            x = self.ipl(x)
            x = x * 2 - 1
            x = self.hpl(x)
            x = self.opl(x)
            # x = torch.einsum("ij,pj->pi", self.scale, x)
            return x

    class autoCluster(nn.Module):
        def normDist(self, x, p):
            return torch.norm((x - p), dim=-1)

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
                [self.normDist(x, self.pointPose[i, :]) ** 2 for i in range(self.k)],
                dim=1,
            )
            W = self.FUN(self.pointWeight)
            # W = (W - 1) ** 3 + 1
            # Wei = torch.sum(self.FUN(self.pointWeight), dim=0)
            imd = torch.einsum("ij,ij->ij", D, W)
            sum = torch.sum(imd, dim=0)
            # print(sum, imd)
            return sum

    def parametrize(self, N: int = 10000) -> None:
        encoderOptimizer: optim.Adam = optim.Adam(self.encoder.parameters(), lr=0.001)
        decoderOptimizer: optim.Adam = optim.Adam(self.decoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        while criterion(OUT, self.dataSet) > 1e-6:
            encoderOptimizer.zero_grad()
            decoderOptimizer.zero_grad()

            P = self.encoder(self.dataSet)
            OUT = self.decoder(P)
            Loss: torch.Tensor = criterion(OUT, self.dataSet)
            print(Loss)
            Loss.backward()

            encoderOptimizer.step()
            decoderOptimizer.step()
        self.parameter = self.encoder(self.dataSet).detach().numpy()

    def parametrizeWithCluster(self, N: int = 10000, k: int = 3) -> None:
        P = self.encoder(self.dataSet)
        self.KMEAN = self.autoCluster(P, k)
        encoderOptimizer: optim.Adam = optim.Adam(self.encoder.parameters(), lr=0.001)
        decoderOptimizer: optim.Adam = optim.Adam(self.decoder.parameters(), lr=0.001)
        KMEANOptimizer: optim.Adam = optim.Adam(self.KMEAN.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        _ = 1
        CT = 100
        # st = False
        while CT > 1e-6:
            encoderOptimizer.zero_grad()
            decoderOptimizer.zero_grad()
            # KMEANOptimizer.zero_grad()

            P = self.encoder(self.dataSet)
            # XSCALE = torch.max(P[:, 0]) - torch.min(P[:, 0])
            # YSCALE = (
            #     torch.kthvalue(P[:, 1], int(len(P[:, 1]) * 0.75)).values
            #     - torch.kthvalue(P[:, 1], int(len(P[:, 1]) * 0.25)).values
            # )
            # ans = torch.sum(self.KMEAN(P[:, -1:]))
            OUT = self.decoder(P)
            CT = criterion(OUT, self.dataSet)
            Loss: torch.Tensor = CT
            # if ans < 5e-5 and CT < 5e-5:
            #     break
            if CT < 5e-4:
                #     st = True
                # if st:
                i = 1
                while True:
                    KMEANOptimizer.zero_grad()
                    encoderOptimizer.zero_grad()
                    decoderOptimizer.zero_grad()
                    P = self.encoder(self.dataSet)
                    OUT = self.decoder(P)
                    CT = criterion(OUT, self.dataSet)
                    ans = torch.sum(self.KMEAN(P))
                    Loss = ans
                    Loss.backward()
                    print("i", i, ans)
                    if ans < 5e-4:
                        break
                    KMEANOptimizer.step()
                    encoderOptimizer.step()
                    decoderOptimizer.step()
                    i += 1
                continue
            # print(_, Loss, CT)
            Loss.backward()
            if _ % 1000 == 0:
                print(_, Loss, CT)
                # # break
                self.parameter = self.encoder(self.dataSet)
                self.output = OUT
                OP = self.KMEAN.pointPose
                print(OP)
                self.drawParameter(_)
                self.drawOutput(_)

            # KMEANOptimizer.step()
            encoderOptimizer.step()
            decoderOptimizer.step()
            _ += 1
        self.parameter = self.encoder(self.dataSet)
        self.output = OUT


if __name__ == "__main__":
    test1 = sineTest()
    test1.generateData()
    test1.drawData()
    # test1.parametrize()
    # test1.drawParameter(1)
    test1.parametrizeWithCluster()
    test1.drawParameter(2)
