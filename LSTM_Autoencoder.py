from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense,TimeDistributed, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K
import os, pickle



class LSTM_Autoencoder:
    def __init__(self, input_shape, lstm_dims, latent_space_dim):
        self.input_shape = input_shape  # dimension of input data: frequency-bins, time-windows, amplitude
        self.latent_space_dim = latent_space_dim
        self.lstm_dims = lstm_dims

        self.encoder = None
        self.decoder = None
        self.model = None
        self.model_input = None


        self._build()

    def _build(self):
        inputs = Input(shape=self.input_shape, name="encoder_input")
        encoded = LSTM(self.latent_space_dim, name="lstm_1")(inputs)
        encoded = Lambda(self.repeat_vector, output_shape=(None, self.latent_space_dim), name="RepeatVector")([encoded, inputs])
        self.encoder = Model(inputs, encoded, name="encoder")

        decoder_input = Input(shape=(None,self.latent_space_dim), name="decoder_input")
        decoded = LSTM(self.latent_space_dim, return_sequences=True, name="lstm_decoder_1")(decoder_input)
        dense = TimeDistributed(Dense(self.input_shape[1]), name="decoder_output")(decoded)
        self.decoder = Model(decoder_input, dense, name="decoder")

        model_input = inputs
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="Autoencoder")

    def repeat_vector(self, args):
        layer_to_repeat = args[0]
        sequence_layer = args[1]
        return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)


        # self._build_encoder()
        # self._build_decoder()
        # self._build_autoencoder()

    # def _build_encoder(self):
    #     encoder_input = Input(shape=self.input_shape, name="Encoder_Input")
    #     self.model_input = encoder_input
    #     if(len(self.lstm_dims) > 0):
    #         lstm_layers = self._add_lstm_layers(encoder_input)
    #     else:
    #         lstm_layers = encoder_input
    #     latent_space_lstm = LSTM(self.latent_space_dim, activation='relu')(lstm_layers)
    #     self.encoder = Model(encoder_input, latent_space_lstm)
    #
    #
    # def _add_lstm_layers(self, encoder_input):
    #     x = LSTM(self.lstm_dims[0], activation='relu', name=f"lstm_{0}")(encoder_input)
    #
    #     for i, lstm_dim in enumerate(self.lstm_dims):
    #         if i > 0:
    #             x = LSTM(lstm_dim, activation='relu', name=f"lstm_{i}", return_sequences=True)(x)
    #     return x
    #
    # def _build_decoder(self):
    #     decoder_input = Input(shape=self.latent_space_dim, name="Decoder_Input")
    #     repeat_vector = Lambda(self.repeat_vector, output_shape=(None, self.latent_space_dim))([decoder_input, self.model_input])
    #     if (len(self.lstm_dims) > 0):
    #         lstm_layers = self._add_decoder_lstms(repeat_vector)
    #     else:
    #         lstm_layers = repeat_vector
    #     last_lstm = LSTM(self.latent_space_dim, activation='relu', return_sequences=True)(lstm_layers)
    #     dense = TimeDistributed(Dense(self.input_shape[1]))(last_lstm)
    #     self.decoder = Model(decoder_input, dense)
    #
    #
    # def _add_decoder_lstms(self, decoder_input):
    #     x = decoder_input
    #
    #     for lstm_dim in reversed(self.lstm_dims):  # go backwards trough layers. Ignore the first layer, because we don't need ReLU or BN on it
    #         x = LSTM(lstm_dim, activation='relu', return_sequences=True)(x)
    #     return x
    #
    # def _build_autoencoder(self):
    #     model_input = self.model_input
    #     model_output = self.decoder(self.encoder(model_input))
    #     self.model = Model(model_input, model_output, name="Autoencoder")

    def compile_model(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer,
                           loss=mse_loss)

    def train(self, x_train, num_epochs):
        self.model.fit(x_train,
                       x_train,
                       batch_size=1,
                       epochs=num_epochs,
                       shuffle=True)

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
            self.lstm_dims,
            self.latent_space_dim,
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
        autoencoder = LSTM_Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.model.load_weights(weights_path)
        return autoencoder
