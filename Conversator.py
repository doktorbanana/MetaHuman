from Variational_Autoencoder_alla_Valerio import VAE
from Snippets import Snippets
from IPython.display import Audio
import numpy as np
import time


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
        self.name = name

        self.min_size_fraction = min_size_fraction
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sr = sample_rate

        self.snippet_generator = Snippets(path=None,
                                          min_size_fraction=self.min_size_fraction,
                                          win_length=self.win_length,
                                          n_fft=self.n_fft,
                                          hop_length=self.hop_length)
        self.snippet_autoencoder = VAE.load(snippet_model_path)
        self.order_autoencoder = VAE.load(order_model_path)

        self.partner = partner
        self.last_song_heard = None
        self.song_to_sing = None
        self.remembered_spectos = None
        self.currently_singing = False

    """
    ========
      SING
    ========
    Here we want the model to generate a new song, based on the last song it heard. It should output the PCA-data and
    send it to its partner.
    """

    def sing(self):
        self.currently_singing = True
        print(self.name + ": LaLeLu... I'm singing a machine Song. DubiSchubiDu...")
        self.partner.last_song_heard = self.song_to_sing
        Audio(self.song_to_sing, autoplay=True, rate=self.sr)
        duration = self.song_to_sing.shape[0] / self.sr
        start_time = time.time()
        if time.time() > start_time + duration + 1:
            print(self.name + ": I'm done singing my machine song. Did you like it?")
            self.currently_singing = False

    def come_up_with_something(self):
        pass

    """
    ========
     Listen
    ========
    Here we want the model to prepare a new song, based on the song it just received.
    """

    def listen(self):
        print(self.name + ": Ohhuu... I'm listening to nice machine music...")
        self.song_to_sing = self.pcm_to_pcm(self.last_song_heard)

    def pcm_to_pcm(self, pcm_data):
        spectos = self.pcm_to_spectos(pcm_data)
        new_pcm = Snippets.specto_to_pcm(model=self.snippet_autoencoder,
                                         data=spectos,
                                         hop_length=self.hop_length,
                                         n_fft=self.hop_length,
                                         win_length=self.win_length)
        return new_pcm

    def pcm_to_spectos(self, pcm_data):
        self.snippet_generator.data = pcm_data
        spectos = self.snippet_generator.get_snippet_spectos()
        self.remembered_spectos.extend(spectos)
        return spectos

    """
    ==============
        Dream
    ==============
    Here we want the model to add the heard songs to it's training data.
    """

    def dream(self, batch_size=1, epochs=20):
        print(self.name + ": ZzzZzz I'm dreaming about all the music i've heard...")

        # Retrain the Snippet-Autoencoder
        x_train = np.asarray(self.remembered_spectos)
        self.snippet_autoencoder.train(x_train, x_train, batch_size=batch_size, num_epochs=epochs)

        # Retrain the Song_Order_Autoencoder
