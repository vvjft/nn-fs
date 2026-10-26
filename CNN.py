"""A convolutional network followed by fully connected MLP layers."""

import numpy as np

from MLP import MLP, activation_functions


class CNN(MLP):
    """Stack valid/same convolutions and an activation after each one.

    ``layers`` contains channel counts, including the input channel count,
    e.g. ``[1, 8, 16]``. ``dense_layers`` contains the sizes of the fully
    connected layers after flattening the final feature maps, e.g. ``[64, 10]``.
    ``input_shape`` is channels-first, e.g. ``(1, 28, 28)``.
    ``feedforward`` accepts a single image, a batch in NCHW format, or the
    MLP-style flattened batch ``(pixels, samples)``. A single image produces
    one output per dense output unit; flattened input produces ``(outputs, samples)``.
    ``forward_layers`` returns the feature maps from each convolution as NCHW
    arrays, which is useful for visualizing intermediate layers.

    The convolutional and fully connected forward passes are implemented.
    Training still needs a CNN-specific implementation.
    """

    def __init__(
        self,
        layers,
        dense_layers=(64, 10),
        input_shape=(1, 28, 28),
        kernel_size=3,
        stride=1,
        padding="valid",
        cost_function="cross_entropy",
        activation_function="sigmoid",
        output_activation_function="softmax",
        dropout_rate=0.3,
    ):

        self.input_shape = tuple(input_shape)
        self.kernel_size = self._pair(kernel_size, "kernel_size")
        self.stride = int(stride)
        self.padding = padding
        self.conv_layers = tuple(layers)
        self.dense_layers = tuple(dense_layers)
        self.activation_name = activation_function
        self.output_activation_name = output_activation_function

        self.feature_shape = self._get_feature_shape()
        flattened_size = int(np.prod(self.feature_shape))

        super().__init__(
            layers=[flattened_size, *self.dense_layers],
            cost_function=cost_function,
            activation_function=activation_function,
            dropout_rate=dropout_rate,
        )
        self.W = [
            np.random.randn(out_size, in_size) / np.sqrt(in_size)
            for in_size, out_size in zip(self.L[:-1], self.L[1:])
        ]
        self.B = [np.zeros((out_size, 1)) for out_size in self.L[1:]]

        self.conv_W = [
            np.random.randn(out_channels, in_channels, *self.kernel_size)
            / np.sqrt(in_channels * self.kernel_size[0] * self.kernel_size[1])
            for in_channels, out_channels in zip(layers[:-1], layers[1:])
        ]
        self.conv_B = [np.zeros((out_channels, 1)) for out_channels in layers[1:]]

        if output_activation_function is None:
            self.output_activation_function = self.activation_function
        else:
            self.output_activation_function = activation_functions[output_activation_function][0]

    @staticmethod
    def _pair(value, name):
        if isinstance(value, int):
            pair = (value, value)
        else:
            pair = tuple(value)
        if len(pair) != 2 or any(size <= 0 for size in pair):
            raise ValueError(f"{name} must be a positive integer or a pair of positive integers.")
        return tuple(int(size) for size in pair)

    def _as_batch(self, X):
        X = np.asarray(X)
        input_channels, height, width = self.input_shape

        if X.ndim == 1 and X.size == input_channels * height * width:
            return X.reshape((1, *self.input_shape)), "image"
        if X.ndim == 2 and X.shape == (height, width) and input_channels == 1:
            return X.reshape((1, 1, height, width)), "image"
        if X.ndim == 2 and X.shape[0] == input_channels * height * width:
            return X.T.reshape((-1, *self.input_shape)), "mlp"
        if X.ndim == 3 and X.shape == self.input_shape:
            return X[np.newaxis, ...], "image"
        if X.ndim == 3 and input_channels == 1 and X.shape[1:] == (height, width):
            return X[:, np.newaxis, ...], "batch"
        if X.ndim == 4:
            return X, "batch"

        raise ValueError(
            "Input must be a single image, an NCHW batch, or a flattened "
            "MLP-style array with shape (pixels, samples)."
        )

    def _get_feature_shape(self):
        channels, height, width = self.input_shape
        kernel_height, kernel_width = self.kernel_size
        for _ in self.conv_layers[1:]:
            if self.padding == "same":
                height = int(np.ceil(height / self.stride))
                width = int(np.ceil(width / self.stride))
            else:
                if kernel_height > height or kernel_width > width:
                    raise ValueError("Kernel dimensions cannot exceed the input dimensions with valid padding.")
                height = (height - kernel_height) // self.stride + 1
                width = (width - kernel_width) // self.stride + 1
        return (self.conv_layers[-1], height, width)

    @staticmethod
    def _softmax(z, axis=1):
        shifted = z - np.max(z, axis=axis, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

    def _activate_conv(self, z):
        if self.activation_name == "softmax":
            return self._softmax(z, axis=1)
        return self.activation_function(z)

    def _activate_dense_hidden(self, z):
        if self.activation_name == "softmax":
            return self._softmax(z, axis=1)
        return self.activation_function(z)

    def _activate_dense_output(self, z):
        if self.output_activation_name == "softmax":
            return self._softmax(z, axis=1)
        return self.output_activation_function(z)

    def _convolve(self, X, kernels, biases):
        kernel_height, kernel_width = self.kernel_size

        if X.shape[1] != kernels.shape[1]:
            raise ValueError(
                f"Input has {X.shape[1]} channels, but this convolution expects {kernels.shape[1]}."
            )

        if self.padding == "same":
            height, width = X.shape[-2:]
            out_height = int(np.ceil(height / self.stride))
            out_width = int(np.ceil(width / self.stride))
            pad_height = max((out_height - 1) * self.stride + kernel_height - height, 0)
            pad_width = max((out_width - 1) * self.stride + kernel_width - width, 0)
            X = np.pad(
                X,
                (
                    (0, 0),
                    (0, 0),
                    (pad_height // 2, pad_height - pad_height // 2),
                    (pad_width // 2, pad_width - pad_width // 2),
                ),
            )
        elif kernel_height > X.shape[-2] or kernel_width > X.shape[-1]:
            raise ValueError("Kernel dimensions cannot exceed the input dimensions with valid padding.")

        windows = np.lib.stride_tricks.sliding_window_view(
            X, (kernel_height, kernel_width), axis=(-2, -1)
        )
        windows = windows[:, :, :: self.stride, :: self.stride, :, :]
        convolved = np.einsum("nchwkl,ockl->nohw", windows, kernels, optimize=True)
        return convolved + biases.reshape(1, -1, 1, 1)

    def _forward_convolutions(self, batch):
        if batch.shape[1:] != self.input_shape:
            raise ValueError(
                f"Expected each image to have shape {self.input_shape}, got {batch.shape[1:]} instead."
            )

        activations = batch
        layer_outputs = []
        for kernels, biases in zip(self.conv_W, self.conv_B):
            activations = self._activate_conv(self._convolve(activations, kernels, biases))
            layer_outputs.append(activations)
        return layer_outputs, activations

    def _forward_batch(self, batch):
        layer_outputs, conv_output = self._forward_convolutions(batch)
        dense_output = conv_output.reshape(conv_output.shape[0], -1)
        for layer_index, (weights, biases) in enumerate(zip(self.W, self.B)):
            dense_output = dense_output @ weights.T + biases.T
            if layer_index == len(self.W) - 1:
                dense_output = self._activate_dense_output(dense_output)
            else:
                dense_output = self._activate_dense_hidden(dense_output)
        return layer_outputs, dense_output

    def forward_layers(self, X):
        """Return the NCHW feature maps produced by each convolution layer."""
        batch, _ = self._as_batch(X)
        layer_outputs, _ = self._forward_convolutions(batch)
        return layer_outputs

    def feedforward(self, X):
        batch, input_format = self._as_batch(X)
        _, output = self._forward_batch(batch)

        if input_format == "image":
            return output[0]
        if input_format == "mlp":
            return output.T
        return output

    def fit(self, *args, **kwargs):
        raise NotImplementedError("CNN training has not been implemented yet.")

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("CNN classification evaluation has not been implemented yet.")

    def save(self, path):
        """Save dense parameters via MLP and convolution parameters separately."""
        super().save(path)
        np.savez(f"{path}/conv_weights.npz", *self.conv_W)
        np.savez(f"{path}/conv_biases.npz", *self.conv_B)

    def load(self, path):
        """Load both dense and convolution parameters for this architecture."""
        super().load(path)
        conv_weights = np.load(f"{path}/conv_weights.npz")
        conv_biases = np.load(f"{path}/conv_biases.npz")
        self.conv_W = [conv_weights[key] for key in conv_weights.files]
        self.conv_B = [conv_biases[key] for key in conv_biases.files]
