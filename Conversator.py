from Variational_Autoencoder_alla_Valerio import VAE
from Snippets import Snippets

class Conversator:
    def __init__(self, snippet_model_path, order_model_path, partner, hop_length, n_fft, win_length):
        self.snippet_autoencoder = VAE.load(snippet_model_path)
        self.order_autoencoder = VAE.load(order_model_path)
        self.partner = partner
        self.last_song_heard = None

    """
    ============
    SING
    ============
    Here we want the model to generate a new song, based on the last song it heard. It should output the PCA-data and
    send it to its partner.
    """
    def sing(self, song_order):
        pca_data =
    

        pass

    def listen(self):
        pass

    def dream(self):
        pass