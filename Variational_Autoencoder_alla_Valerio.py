from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, \
    Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

"""
-------------------------------------------
AUTOENCODER CLASS
-------------------------------------------
"""


class VAE:

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim, num_of_train_data=0):
        self.input_shape = input_shape  # dimension of input data: frequency-bins, time-windows, amplitude
        self.conv_filters = conv_filters  # a list with the number of filters per layer
        self.conv_kernels = conv_kernels  # a list with the kernel size per layer
        self.conv_strides = conv_strides  # a list with the strides per layer
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = 1000000
        self._shape_before_bottleneck = None

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
        encoder_input = Input(shape=self.input_shape, name="encoder_input")
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

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
            name=f"encoder_conv_layer_{layer_index + 1}")
        x = conv_layer(x)  # apply the new conv to x -> Convolution with kernels give multiple 2D Arrays
        x = ReLU(name=f"encoder_relu_{layer_index + 1}")(x)  # apply a ReLU activation to x
        x = BatchNormalization(name=f"encoder_bn_{layer_index + 1}")(
            x)  # apply Batch Normalization to x
                # (less overfitting-problems, no vanishing Gradient or exploding Gradient)
        return x



    def _add_bottleneck(self, x):
        self._shape_before_bottleneck = K.int_shape(x)[1:]  # Ignore the first dim, which is the batch size
        x = Flatten(name="encoder_flatten")(x)  # Flatten Data

        # Gaussian Sampling: Sample a point in the gaussian distribution from a point in standard normal distribution
        self.mu = Dense(self.latent_space_dim, name="min_vector_mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)

        """
        def sample_point_from_normal_dist(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.latent_space_dim), mean=0., stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        x = Lambda(sample_point_from_normal_dist,
                   name="encoder_output")([self.mu, self.log_variance])
        """
        x = LambdaLayer(self.latent_space_dim)([self.mu, self.log_variance])
        return x


    """
    ------------
    Decoder Part 
    ------------
    """

    def _build_decoder(self):
        decoder_input = Input(shape=self.latent_space_dim, name="decoder_input")
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = Reshape(self._shape_before_bottleneck, name="decoder_reshape_layer")(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(
            self._shape_before_bottleneck)  # Product of the dimensions before the latent space
        # -> Size of the flattened data before Latent Space
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_conv_transpose_layers(self, x):
        for layer_index in reversed(range(1,
                                          self._num_conv_layers)):  # go backwards trough layers.
            # Ignore the first layer, because we don't need
            # ReLU or BN on it
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers - layer_index}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{self._num_conv_layers - layer_index}")(x)
        x = BatchNormalization(name=f"decoder_bn_{self._num_conv_layers - layer_index}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=self.input_shape[-1],  # We want to recreate the input shape
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="decoder_output_sigmoid")(x)
        return output_layer

    """
    ------------
    VAE Part 
    ------------
    """
    def compile_model(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self._calculate_combined_loss,
                           metrics=[self.calculate_reconstruction_loss])

    def _calculate_combined_loss(self, y_true, y_pred):
        reconstruction_loss = self.calculate_reconstruction_loss(y_true, y_pred)
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=1)
        loss = (self.reconstruction_loss_weight * reconstruction_loss) + kl_loss

        return loss

    @staticmethod
    def calculate_reconstruction_loss(y_true, y_pred):
        # Mean Squared Error
        error = y_true - y_pred
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])  # mean squared error
        return reconstruction_loss



    def train(self, x_train, batch_size, num_epochs):
        self.num_of_train_data += x_train.shape[0]
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True
                       )

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="Valerio")

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
        self.model.save(save_folder + "\\" + "model.h5") # recursion error. too many trainable parameters?

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
        # Load the parameters:
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        loaded_autoencoder = VAE(*parameters)

        weights_path = os.path.join(save_folder, "weights.h5")
        loaded_autoencoder.model.load_weights(weights_path)
        """
        #Load Model:
        model_path = os.path.join(save_folder, "model.h5")
        loaded_autoencoder.model = load_model(model_path, custom_objects={'LambdaLayer': LambdaLayer,
                                                                    '_calculate_combined_loss': cls._calculate_combined_loss,
                                                                    'calculate_reconstruction_loss': cls.calculate_reconstruction_loss}
                                                                    )

        print(f"This model was trained with {loaded_autoencoder.num_of_train_data} wavesets "
              f"and has a {loaded_autoencoder.latent_space_dim}D latent space")
        """
        return loaded_autoencoder


"""
-------------
Custom Layer
-------------
"""
class LambdaLayer(tf.keras.layers.Layer):
    def __init__(self, latent_space_dim):
        super(LambdaLayer, self).__init__()
        self.latent_space_dim = latent_space_dim

    def call(self, inputs):
        mu, log_variance = inputs
        epsilon = K.random_normal(shape=K.shape(self.latent_space_dim), mean=0., stddev=1.)
        sampled_point = mu + K.exp(log_variance / 2) * epsilon
        return sampled_point

    def get_config(self):
        return {'latent_space_dim': self.latent_space_dim}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":

    autoencoder = None

    for i in range(3):

        num = (i+1) * 500
        """
        -----------------
        Loading the Data
        -----------------
        """

        subfolder = "0.25_16"
        x_train = np.load("data_and_models\\" + subfolder + "\\spectos" + str(num) + ".npy")

        """
        ------------------------
        Buildung the VAE
        ------------------------
        """

        if autoencoder is None:
            autoencoder = VAE(
                input_shape=(x_train[0].shape[0], x_train[0].shape[1], x_train[0].shape[2]),
                conv_filters=(512, 256, 128, 64, 32),
                conv_kernels=(3, 3, 3, 3, 3),
                conv_strides=(2, 2, 2, 2, (2,1)),
                latent_space_dim=128
            )
            autoencoder.summary()

            LEARNING_RATE = 0.0005
            BATCH_SIZE = 256
            EPOCHS = 20

            autoencoder.compile_model(LEARNING_RATE)

        print(x_train.shape)

        """
        ---------------------
        Train the VAE
        ---------------------
        """

        autoencoder.train(x_train, BATCH_SIZE, EPOCHS)

        """
        ----------------
        Save VAE
        ----------------
        """

        save_path = "data_and_models\\" + subfolder + "\\Valerio_" + str(autoencoder.latent_space_dim) + "D_" + str(autoencoder.num_of_train_data) + "samples_" + str(EPOCHS) + "Epochs"
        autoencoder.save(save_path)
        #test_load = VAE.load(save_path)
        #test_load.summary()

