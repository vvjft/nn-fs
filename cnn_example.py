"""Visualize one MNIST image before and after the CNN forward pass."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from CNN import CNN
from data_loader import mnist_loader


def main():
    parser = argparse.ArgumentParser(description="Show convolution outputs for one MNIST image.")
    parser.add_argument("--data-path", default="./data", help="Folder containing the MNIST data.")
    parser.add_argument("--index", type=int, default=0, help="Image index in the test set.")
    args = parser.parse_args()

    dataset = mnist_loader(download=False, path=args.data_path)
    X_test, Y_test = dataset.test
    if not 0 <= args.index < X_test.shape[1]:
        raise IndexError(f"Image index must be between 0 and {X_test.shape[1] - 1}.")

    image = X_test[:, args.index].reshape(28, 28)
    digit = int(np.argmax(Y_test[:, args.index]))

    # Two convolution layers: 1 input channel -> 4 -> 8 feature maps.
    cnn = CNN(
        layers=[1, 4, 8],
        dense_layers=[64, 10],
        input_shape=(1, 28, 28),
        kernel_size=3,
        stride=1,
        padding="valid",
        activation_function="sigmoid",
    )

    layer_outputs = cnn.forward_layers(image)
    output = cnn.feedforward(image)
    predicted_digit = int(np.argmax(output))

    print(f"Rzeczywista cyfra: {digit}")
    print(f"Wynik feedforward (10 klas): {output}")
    print(f"Klasa z najwyższym wynikiem: {predicted_digit}")

    maps_to_show = 4
    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = figure.add_gridspec(3, maps_to_show, height_ratios=(1, 1.5, 1))

    input_ax = figure.add_subplot(grid[0, 0])
    input_ax.imshow(image, cmap="gray")
    input_ax.set_title(f"Obraz wejściowy: {digit}")
    input_ax.axis("off")

    for channel in range(maps_to_show):
        ax = figure.add_subplot(grid[1, channel])
        if channel < layer_outputs[0].shape[1]:
            ax.imshow(layer_outputs[0][0, channel], cmap="viridis")
            ax.set_title(f"Po konwolucji — filtr {channel + 1}")
        ax.axis("off")

    output_ax = figure.add_subplot(grid[2, :])
    output_ax.bar(np.arange(output.size), output)
    output_ax.set_xticks(np.arange(output.size))
    output_ax.set_xlabel("Cyfra")
    output_ax.set_ylabel("Prawdopodobieństwo")
    output_ax.set_title(f"Output po feedforward — klasa {predicted_digit}")
    figure.suptitle("CNN z warstwami konwolucyjnymi i MLP")
    output_path = "cnn_example.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Zapisano wykres: {output_path}")


if __name__ == "__main__":
    main()
