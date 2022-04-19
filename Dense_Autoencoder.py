from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Masking, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import pickle
from Snippets import Snippets
import math

"""
-------------------------------------------
DENSE AUTOENCODER CLASS
-------------------------------------------
"""


class Dense_Autoencoder:

    def __init__(self, input_shape, num_of_neurons, latent_space_dim, num_of_train_data=None, mask_value=0, old_min=0,
                 old_max=0,
                 _latent_x_max=None,
                 _latent_x_min=None,
                 _latent_y_max=None,
                 _latent_y_min=None):
        self.input_shape = input_shape  # dimension of input data: frequency-bins, time-windows, amplitude
        self.num_of_neurons = num_of_neurons
        self.latent_space_dim = latent_space_dim
        self._shape_before_bottleneck = None
        self.mask_value = mask_value

        self.encoder = None
        self.decoder = None
        self.model = None
        self.model_input = None
        self.num_of_train_data = num_of_train_data

        self.old_min = old_min
        self.old_max = old_max

        self.latent_x_max = _latent_x_max
        self.latent_x_min = _latent_x_min
        self.latent_y_max = _latent_y_max
        self.latent_y_min = _latent_y_min

        self.num_of_dense_layers = len(self.num_of_neurons)
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
        masking = Masking(mask_value=self.mask_value)(encoder_input)
        dense_layers = self._add_dense_layers(masking)
        bottleneck = self._add_bottleneck(dense_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="Encoder")

    def _add_dense_layers(self, encoder_input):
        x = encoder_input
        for i, num in enumerate(self.num_of_neurons):
            x = Dense(num, activation='relu', name=f"Dense_Layer_{i}")(x)
        return x

    def _add_bottleneck(self, x):
        x = Dense(self.latent_space_dim, name="Encoder_Output")(x)
        return x

    """
    ------------
    Decoder Part 
    ------------
    """

    def _build_decoder(self):
        decoder_input = Input(shape=self.latent_space_dim, name="Decoder_Input")
        dense_layers = self._add_transpose_dense_layers(decoder_input)
        decoder_output = self._add_decoder_output(dense_layers)
        self.decoder = Model(decoder_input, decoder_output, name="Decoder")

    def _add_transpose_dense_layers(self, x):
        for i in reversed(range(self.num_of_dense_layers)):  # go backwards trough layers.
            num = self.num_of_neurons[-i]
            x = Dense(num, activation='relu', name=f"Decoder_Dense_{i}")(x)
        return x

    def _add_decoder_output(self, x):
        output_layer = Dense(self.input_shape, activation='sigmoid')(x)
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
        self.model.compile(optimizer=optimizer, loss='mse')

    def train(self, x_train, x_test, batch_size, num_epochs):
        self.num_of_train_data = x_train.shape[0]
        hist = self.model.fit(x_train,
                              x_train,
                              batch_size=batch_size,
                              epochs=num_epochs,
                              shuffle=True,
                              validation_data=(x_test, x_test))
        return hist

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
        self._save_optimizer_state(save_folder)
        model_save_path = os.path.join(save_folder, "model.h5")
        self.model.save(model_save_path)

    def _create_folder(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.num_of_neurons,
            self.latent_space_dim,
            self.num_of_train_data,
            self.mask_value,
            self.old_min,
            self.old_max,
            self.latent_x_max,
            self.latent_x_min,
            self.latent_y_max,
            self.latent_y_min]

        save_path = os.path.join(save_folder, "parameters.pkl")

        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _save_optimizer_state(self, save_folder):
        # Save optimizer weights.
        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        optimizer_path = os.path.join(save_folder, "optimizer.pkl")
        with open(optimizer_path, 'wb') as f:
            pickle.dump(weight_values, f)

    @classmethod
    def load(cls, save_folder=".", learning_rate=0.0001):
        # Load the parameters:
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        loaded_autoencoder = Dense_Autoencoder(*parameters)

        # Compile the model (to be able to retrain it with additional data if needed)
        loaded_autoencoder.compile_model(learning_rate=learning_rate)

        # Load the Weights
        weights_path = os.path.join(save_folder, "weights.h5")
        loaded_autoencoder.model.load_weights(weights_path)

        # Load the Optimizer State
        loaded_autoencoder.model._make_train_function()

        optimizer_path = os.path.join(save_folder, "optimizer.pkl")
        with open(optimizer_path, 'rb') as f:
            weight_values = pickle.load(f)
        loaded_autoencoder.model.optimizer.set_weights(weight_values)

        return loaded_autoencoder

