import os
from Variational_Autoencoder_alla_Valerio import VAE
from Dense_Autoencoder import Dense_Autoencoder
from Processing_Communicator import Communicator
from Snippets import Snippets
import sounddevice as sd
import numpy as np
import time
import simpleaudio as sa


class Conversator:
    def __init__(self, snippet_model_path,
                 order_model_path,
                 partner,
                 min_size_fraction,
                 hop_length,
                 n_fft,
                 win_length,
                 sample_rate=44100,
                 name="Valerio",
                 communicator=None):
        self.name = name

        self.min_size_fraction = min_size_fraction
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sr = sample_rate

        self.snippet_autoencoder = VAE.load(snippet_model_path)

        if order_model_path is not None:
            self.order_autoencoder = Dense_Autoencoder.load(order_model_path)
        else:
            self.order_autoencoder = None

        if communicator is not None:
            self.communicator = communicator
        else:
            self.communicator = Communicator()

        self.partner = partner
        self.last_song_heard = None
        self.last_song_imitation = None
        self.last_song_variation = None

        self.remembered_spectos = []
        self.currently_singing = False
        self.start_time = 0
        self.current_song_duration = 0

    """
    ========
      SING
    ========
    Here we want the model to generate a new song, based on the last song it heard. It should output the PCA-data and
    send it to its partner.
    """

    def sing(self, song):
        self.currently_singing = True
        self.partner.last_song_heard = song
        #display(Audio(song, autoplay=True, rate=self.sr))
        #sd.play(song, self.sr)
        audio = song*32767
        audio = audio.astype(np.int16)
        sa.play_buffer(audio, 1, 2, self.sr)
        self.current_song_duration = song.shape[0] / self.sr
        self.start_time = time.time()

    def sing_last_heard_song(self):
        print(self.name + ": Let me try to sing, what i just heard. DubiSchubiDu...")
        self.communicator.send("/" + self.name, "Let me try to sing, what i just heard. DubiSchubiDu...")
        self.sing(self.last_song_imitation)

    def sing_human_song(self, song_orders_path, start_in_secs, end_in_secs):
        print(self.name + ": Let me remember that human song...")
        self.communicator.send("/" + self.name, "Let me remember that human song...")
        self.communicator.send("/" + self.name + "Band","")
        song_orders = np.load(song_orders_path, allow_pickle=True)

        human_song = song_orders[np.random.randint(0, song_orders.shape[0])]
        human_song = human_song[int(start_in_secs * self.min_size_fraction):int(end_in_secs * self.min_size_fraction)]

        print("Song Shape: " + str(human_song.shape))

        human_pcm, _ = Snippets.latent_representation_to_pcm(latent_representations=human_song,
                                                             model=self.snippet_autoencoder,
                                                             hop_length=self.hop_length,
                                                             n_fft=self.n_fft,
                                                             win_length=self.win_length)
        print(self.name + ": LaLeLu... I'm singing one of the human songs...")
        self.communicator.send("/" + self.name, "LaLeLu... I'm singing one of the human songs...")
        self.sing(human_pcm)

    def sing_machine_song(self, max_length_in_seconds=20):
        print(self.name + ": Let me compose a new song...")
        self.communicator.send("/" + self.name, "Let me compose a new song...")

        random_x = ((np.random.rand() * 2) - 1) * (
                    self.order_autoencoder.latent_x_max - self.order_autoencoder.latent_x_min)
        random_y = ((np.random.rand() * 2) - 1) * (
                    self.order_autoencoder.latent_y_max - self.order_autoencoder.latent_y_min)
        random_dot = np.asarray([random_x, random_y]).reshape(1, 2)

        song_order = self.order_autoencoder.decoder.predict(random_dot)

        reshaped_order = song_order.reshape(int(song_order.shape[1] / 128), 128)
        denorm_order = Snippets._denormalise(reshaped_order, 0.001, 1, self.order_autoencoder.old_min,
                                             self.order_autoencoder.old_max)
        pad_prediction = np.argwhere(denorm_order < (self.order_autoencoder.old_min + (
                    abs(self.order_autoencoder.old_min - self.order_autoencoder.old_max) * 0.25)))[:,
                         0].min()  # predict which coordinates were probably just filled up
        cut_reconstructed_data = denorm_order[:pad_prediction]

        cut_reconstructed_data = cut_reconstructed_data[:int(max_length_in_seconds * self.min_size_fraction)]

        pcm_data, _ = Snippets.latent_representation_to_pcm(latent_representations=cut_reconstructed_data,
                                                            model=self.snippet_autoencoder,
                                                            hop_length=self.hop_length,
                                                            n_fft=self.n_fft,
                                                            win_length=self.win_length)

        print(self.name + ": BriiBrazzzFuaazz... I'm singing a machine song.")
        self.communicator.send("/" + self.name, "BriiBrazzzFuaazz... I'm singing a machine song.")
        self.sing(pcm_data)

    def sing_variation_of_last_heard(self, var_amount=1):
        print(self.name + ": ZiiiSch... I'm singing a variation of the last song.")
        self.communicator.send("/" + self.name, "ZiiiSch... I'm singing a variation of the last song.")

        if self.last_song_variation is not None:
            self.sing(self.last_song_variation)
        else:
            print(self.model_name + ": Upps, i didn't think about a variation.")
            self.communicator.send("/" + self.name, "Upps, i didn't think about a variation.")

    def quiet_please(self):
        while time.time() < self.start_time + self.current_song_duration + 1:
            pass
        self.check_end_of_singing()

    def check_end_of_singing(self):
        if not time.time() < self.start_time + self.current_song_duration + 1:
            print(self.name + ": I'm done singing my song. Did you like it?")
            self.communicator.send("/" + self.name + "Done", "I'm done singing my song. Did you like it?")
            self.currently_singing = False

    """
    ========
     Listen
    ========
    Here we want the model to prepare a new song, based on the song it just received.
    """

    def listen(self):
        print(self.name + ": Nice machine music! I will try to imitate that!")
        self.communicator.send("/" + self.name + "Waits", "Nice machine music! I will try to imitate that!")
        self.last_song_imitation, spectos = Snippets.pcm_to_pcm(model=self.snippet_autoencoder,
                                                                data=self.last_song_heard,
                                                                min_size_fraction=self.min_size_fraction,
                                                                hop_length=self.hop_length,
                                                                n_fft=self.n_fft,
                                                                win_length=self.win_length,
                                                                var_amount=0,
                                                                partner=self.partner)

        if spectos is not None:
            self.remembered_spectos.extend(spectos)
        else:
            print("No spectos... No vocals?")
        return spectos

    def think_about_variation(self, var_amount=1):
        print(self.name + ": Mhhh... Interesting Song! Let me think about a variation...")
        self.communicator.send("/" + self.name + "Waits", "Mhhh... Interesting Song! Let me think about a variation...")
        self.last_song_variation, spectos = Snippets.pcm_to_pcm(model=self.snippet_autoencoder,
                                                                data=self.last_song_heard,
                                                                min_size_fraction=self.min_size_fraction,
                                                                hop_length=self.hop_length,
                                                                n_fft=self.n_fft,
                                                                win_length=self.win_length,
                                                                var_amount=var_amount,
                                                                partner=self.partner)
        if spectos is not None:
            self.remembered_spectos.extend(spectos)
        else:
            print("No spectos... No vocals?")
        return spectos

    """
    ==============
        Dream
    ==============
    Here we want the model to add the heard songs to it's training data.
    """

    def dream(self, batch_size=32, epochs=20, ):
        print(self.name + ": ZzzZzz I'm dreaming about all the music i've heard...")
        self.communicator.send("/" + self.name + "Sleeps", "ZzzZzz I'm dreaming about all the music i've heard...")
        # Retrain the Snippet-Autoencoder
        x_train = np.asarray(self.remembered_spectos)
        self.snippet_autoencoder.train(x_train, x_train, batch_size=batch_size, num_epochs=epochs)
        path = os.path.join("data_and_models", "post_human" + self.name)
        self.snippet_autoencoder.save(path)
        self.remembered_spectos = []
        self.snippet_autoencoder = VAE.load(path=path)


if __name__ =="__main__":

    """ ============ """
    """ Load Models """
    """ ============ """

    valerio_snippet_model_path = os.path.join("data_and_models/2.0_128", "VAE_LedZep128D_25140samples_40Epochs")
    dennis_snippet_model_path = os.path.join("data_and_models/2.0_128", "VAE_beatles128D_50950samples_20Epochs")
    dennis_song_orders_path = os.path.join("data_and_models//2.0_128",
                                           "VAE_beatles128D_50950samples_20Epochs_song_orderbeatles.npy")
    valerio_song_orders_path = os.path.join("data_and_models//2.0_128",
                                           "VAE_LedZep128D_25140samples_40Epochs_song_orderLedZep_40.npy")

    valerio_order_model_path = None
    dennis_order_model_path = None

    communicator = Communicator()

    valerio = Conversator(
        snippet_model_path=valerio_snippet_model_path,
        order_model_path=valerio_order_model_path,
        partner=None,
        min_size_fraction=0.5,
        hop_length=690,
        n_fft=690 * 2,
        win_length=690 * 2,
        sample_rate=44100,
        name="Valerio",
        communicator=communicator)

    dennis = Conversator(
        snippet_model_path=dennis_snippet_model_path,
        order_model_path=dennis_order_model_path,
        partner=valerio,
        min_size_fraction=0.5,
        hop_length=690,
        n_fft=690 * 2,
        win_length=690 * 2,
        sample_rate=44100,
        name="Dennis",
        communicator=communicator)

    valerio.partner = dennis
    dennis.partner = valerio

    """ ============== """
    """ START THE LOOP """
    """ ============== """


    steps = 20
    sub_steps = 1
    count = 1



    for j in range(steps):

        communicator.send("/iteration", str(count))
        print("iteration " + str(count) + " of " + str(steps * sub_steps))

        for i in range(sub_steps):
            print("iteration " + str(count) + " of " + str(steps * sub_steps))
            communicator.send("/iteration", str(count))

            # Start with a human song
            dennis.sing_human_song(dennis_song_orders_path, 20, 40)
            valerio.listen()
            dennis.quiet_please()

            # Answer with an Imitation
            valerio.sing_last_heard_song()
            dennis.think_about_variation(var_amount=0.5)
            valerio.quiet_please()

            # Continue with a Variation
            dennis.sing_variation_of_last_heard()

            # Switch roles
            # Start with a human song
            dennis.quiet_please()
            valerio.sing_human_song(valerio_song_orders_path, 20, 40)
            dennis.listen()
            valerio.quiet_please()

            # Answer with an Imitation
            dennis.sing_last_heard_song()
            valerio.think_about_variation(var_amount=0.5)
            dennis.quiet_please()

            # Continue with a Variation
            valerio.sing_variation_of_last_heard()
            valerio.quiet_please()

            count += 1

        # Retrain Models
        dennis.dream()
        valerio.dream()