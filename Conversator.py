from Variational_Autoencoder_alla_Valerio import VAE
from Snippets import Snippets
from IPython.display import Audio
from librosa import display

class Conversator:
    def __init__(self, snippet_model_path,
                 order_model_path,
                 partner,
                 min_size_fraction,
                 hop_length,
                 n_fft,
                 win_length,
                 sample_rate=44100,
                 name="Valerio"):

        self.snippet_generator = Snippets.snip
        self.snippet_autoencoder = VAE.load(snippet_model_path)
        self.order_autoencoder = VAE.load(order_model_path)
        self.partner = partner
        self.last_song_heard = None
        self.name = name
        self.singing = False
        self.ready_to_sing = False

        self.min_size_fraction = min_size_fraction
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sr = sample_rate
    """
    ============
    SING
    ============
    Here we want the model to generate a new song, based on the last song it heard. It should output the PCA-data and
    send it to its partner.
    """

    def sing(self):
        song = self.prepare_song()
        display(Audio(song, autoplay=True, rate=self.sr))

    def prepare_song(self):
        new_song = self.pcm_to_pcm(self.last_song_heard)
        self.partner.last_song_heard = new_song
        return new_song

    def pcm_to_pcm(self, pcm_data):
        spectos = self.pcm_to_spectos(pcm_data)
        new_pcm = Snippets.spectos_to_pca()
        return latent_representation

    def pcm_to_spectos(self, pcm_data):
        snippet_generator = Snippets(None, self.min_size_fraction, self.win_length, self.n_fft, self.hop_length)
        snippet_generator.data = pcm_data
        spectos = snippet_generator.get_snippet_spectos()
        return spectos

    def latent_represenation_to_pcm(self):


def listen(self):
        pass

    def dream(self):
        pass