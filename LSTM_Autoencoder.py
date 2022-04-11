from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense,TimeDistributed, Masking
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import os, pickle



class LSTM_Autoencoder:
    def __init__(self, input_shape, lstm_dims, latent_space_dim, mask_value):
        self.input_shape = input_shape  # dimension of input data: frequency-bins, time-windows, amplitude
        self.latent_space_dim = latent_space_dim
        self.lstm_dims = lstm_dims
        self.mask_value = mask_value

        self.encoder = None
        self.decoder = None
        self.model = None
        self.model_input = None


        self._build()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = Input(shape=self.input_shape, name="Encoder_Input")
        self.model_input = encoder_input
        masking_layer = self._add_masking_layer(encoder_input)
        if(len(self.lstm_dims) > 0):
            lstm_layers = self._add_lstm_layers(masking_layer)
        else:
            lstm_layers = masking_layer
        latent_space_lstm = LSTM(self.latent_space_dim, activation='relu')(lstm_layers)
        self.encoder = Model(encoder_input, latent_space_lstm)


    def _add_masking_layer(self, encoder_input):
        x = encoder_input
        x = Masking(mask_value=self.mask_value)(x)
        return x

    def _add_lstm_layers(self, encoder_input):
        x = LSTM(self.lstm_dims[0], activation='relu', name=f"lstm_{0}", return_sequences=False)(encoder_input)

        for i, lstm_dim in enumerate(self.lstm_dims):
            if i > 0:
                x = LSTM(lstm_dim, activation='relu', name=f"lstm_{i}", return_sequences=True)(x)
        return x


    def _build_decoder(self):
        decoder_input = Input(shape=self.latent_space_dim, name="Decoder_Input")
        repeat_vector = RepeatVector(self.input_shape[0])(decoder_input)
        lstm_layers = self._add_decoder_lstms(repeat_vector)
        dense = TimeDistributed(Dense(128))(lstm_layers)
        self.decoder = Model(decoder_input, dense)

    def _add_decoder_lstms(self, decoder_input):
        x = decoder_input

        for lstm_dim in reversed(self.lstm_dims):  # go backwards trough layers. Ignore the first layer, because we don't need ReLU or BN on it
            x = LSTM(lstm_dim, activation='relu', return_sequences=True)(x)
        return x

    def _build_autoencoder(self):
        model_input = self.model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="Autoencoder")

    def compile_model(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, x_train, batch_size, num_epochs):
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
