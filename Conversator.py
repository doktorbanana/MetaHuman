from Variational_Autoencoder_alla_Valerio import VAE
from SoundGenerator_new import SoundGenerator

class Conversator:
    def __init__(self, model_path, partner, hop_length, n_fft, win_length):
        self.autoencoder = VAE.load(model_path)
        self.partner = partner
        self.sound_generator = SoundGenerator(self.autoencoder, hop_length=hop_length, win_size=win_length, n_fft=n_fft)
        self.last_song_heard = None

    """
    ============
    SING
    ============
    Here we want the model to generate a new song, based on the last song it heard. It should output the PCA-data and
    send it to its partner.
    """
    def sing(self, song_order):
        pca_data = SoundGenerator.generate_Song()
    

        pass

    def listen(self):
        pass

    def dream(self):
        pass