import numpy as np
import matplotlib.pyplot as plt
import pandas

# --------------------------------
# loading data
from sklearn.datasets import fetch_openml

X, y = fetch_openml("mnist_784", return_X_y=True)
print("SHOW_DATA", X, y, sep="\n")
print("SHOW_DATATYPE", type(X), type(y), sep="\n")
# --------------------------------
n_images = X.shape[0]
# Convert data to image pixel-by-pixel representation
X_images = X.to_numpy().reshape(n_images, 28, 28)

# Flatten the data so that we can apply clustering
X = X_images.reshape(n_images, 28 * 28)
# --------------------------------
# ploting
n_digits = 10
fig, axes = plt.subplots(1, n_digits)
for i in range(n_digits):
    # Get indices of digit i
    digit_i_idx = np.where(y == i)
    print(np.where(y == i))

    # Get images for this digit label
    digit_images = X_images[digit_i_idx]

    # Get an example image for this digit
    image = digit_images[0]

    # Display the image
    axes[i].imshow(image, cmap="grey")

    # Styling: Turn of x/y axis ticks
    axes[i].set_yticks([])
    axes[i].set_xticks([])
# --------------------------------
