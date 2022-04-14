from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Masking, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

"""
-------------------------------------------
AUTOENCODER CLASS
-------------------------------------------
"""


class Autoencoder:

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim, num_of_train_data=None,
                 mask_value=0):
        self.input_shape = input_shape  # dimension of input data: frequency-bins, time-windows, amplitude
        self.conv_filters = conv_filters  # a list with the number of filters per layer
        self.conv_kernels = conv_kernels  # a list with the kernel size per layer
        self.conv_strides = conv_strides  # a list with the strides per layer
        self.latent_space_dim = latent_space_dim
        self._shape_before_bottleneck = None
        self.mask_value = mask_value

        self.encoder = None
        self.decoder = None
        self.model = None
        self.model_input = None
        self.num_of_train_data = num_of_train_data

        self._num_conv_layers = len(conv_filters)

        self._build()


    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    """ 
    ------------
    Encoder Part 
    ------------
    """

    def _build_encoder(self):
        encoder_input = Input(shape=self.input_shape, name="Encoder_Input")
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="Encoder")

    def _add_conv_layers(self, encoder_input):
        x = encoder_input

        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"Encoder_Conv_Layer_{layer_index + 1}")
        x = conv_layer(x)  # apply the new conv to x -> Convolution with kernels give multiple 2D Arrays
        x = ReLU(name=f"Encoder_ReLU_{layer_index + 1}")(x)  # apply a ReLU activation to x
        x = BatchNormalization(name=f"Encoder_BN_{layer_index + 1}")(
            x)  # apply Batch Normalization to x (less overfitting-problems, no vanishing Gradient or exploding Gradient)
        return x

    def _add_bottleneck(self, x):
        self._shape_before_bottleneck = K.int_shape(x)[1:]  # Ignore the first dim, which is the batch size
        x = Flatten(name="Encoder_Flatten")(x)  # Flatten Data
        x = Dense(self.latent_space_dim, name="Encoder_Output")(x)  # Apply a Dense Layer for the latent space to x11
        return x

    """
    ------------
    Decoder Part 
    ------------
    """

    def _build_decoder(self):
        decoder_input = Input(shape=self.latent_space_dim, name="Decoder_Input")
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = Reshape(self._shape_before_bottleneck, name="Decoder_Reshape_Layer")(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="Decoder")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(
            self._shape_before_bottleneck)  # Product of the dimensions before the latent space -> Size of the flattened data before Latent Space
        dense_layer = Dense(num_neurons, name="Decoder_Dense")(decoder_input)
        return dense_layer

    def _add_conv_transpose_layers(self, x):
        for layer_index in reversed(range(1,
                                          self._num_conv_layers)):  # go backwards trough layers. Ignore the first layer, because we don't need ReLU or BN on it
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"Decoder_conv_transpose_layer_{self._num_conv_layers - layer_index}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"Decoder_ReLU_{self._num_conv_layers - layer_index}")(x)
        x = BatchNormalization(name=f"Decoder_BN_{self._num_conv_layers - layer_index}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=self.input_shape[-1],  # We want to recreate the input shape
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"Decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="Decoder_Output_Sigmoid")(x)
        return output_layer

    """
    ------------
    Autoencoder Part 
    ------------
    """

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="Autoencoder")

    def compile_model(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, x_train, batch_size, num_epochs):
        self.num_of_train_data = x_train.shape[0]
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True
                       )

    def summary(self):
        self.model.summary()
        self.encoder.summary()
        self.decoder.summary()

    """
    ------------------
    Saving and Loading
    ------------------
    """

    def save(self, save_folder="."):
        self._create_folder(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def _create_folder(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim,
            self.num_of_train_data
        ]

        save_path = os.path.join(save_folder, "parameters.pkl")

        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.model.load_weights(weights_path)

        print(f"This model was trained with {autoencoder.num_of_train_data} wavesets and has a {autoencoder.latent_space_dim}D latent space")
        return autoencoder


if __name__ == "__main__":
    """
    -----------------
    Loading the Data
    -----------------
    """
    subfolder = "1.0_16"
    x_train = np.load("data_and_models\\" + subfolder + "\\spectos.npy")
    x_train = x_train[:50000]

    """
    ------------------------
    Buildung the Autoencoder
    ------------------------
    """

    autoencoder = Autoencoder(
        input_shape=(x_train[0].shape[0], x_train[0].shape[1], x_train[0].shape[2]),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=128
    )

    autoencoder.summary()

    """
    ---------------------
    Train the Autoencoder
    ---------------------
    """

    LEARNING_RATE = 0.0005
    BATCH_SIZE = 64
    EPOCHS = 15

    autoencoder.compile_model(LEARNING_RATE)
    autoencoder.train(x_train, BATCH_SIZE, EPOCHS)

    """
    ----------------
    Save Autoencoder
    ----------------
    """

    autoencoder.save("Autoencoder_" + str(autoencoder.latent_space_dim) + "D_" + subfolder)


