from sklearn.decomposition import PCA
import numpy as np


def main():
    data = np.array(
        [
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2, 1.6],
            [1, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ]
    )
    print(data)
    pca = PCA(n_components=1)
    pca.fit(data)

    print(pca.components_)
    print(pca.explained_variance_)


if __name__ == "__main__":
    main()
