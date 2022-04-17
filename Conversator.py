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

        self.snippet_autoencoder = VAE.load(snippet_model_path)

        if order_model_path is not None:
            self.order_autoencoder = VAE.load(order_model_path)
        else:
            self.order_autoencoder = None

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
        display(Audio(song, autoplay=True, rate=self.sr))
        self.current_song_duration = song.shape[0] / self.sr
        self.start_time = time.time()

    def sing_last_heard_song(self):
        print(self.name + ": Let me try to sing, what i just heard. DubiSchubiDu...")
        self.sing(self.last_song_imitation)

    def sing_human_song(self, song_orders_path):
        print(self.name + ": Let me remember that human song...")
        song_orders = np.load(song_orders_path, allow_pickle=True)
        human_song = song_orders[np.random.randint(0, song_orders.shape[0])]
        human_song = human_song[:10]
        human_pcm, _ = Snippets.latent_representation_to_pcm(latent_representations=human_song,
                                                             model=self.snippet_autoencoder,
                                                             hop_length=self.hop_length,
                                                             n_fft=self.n_fft,
                                                             win_length=self.win_length)
        print(self.name + ": LaLeLu... I'm singing one of the human songs...")
        self.sing(human_pcm)

    def sing_machine_song(self):
        # HAS TO BE IMPLEMENTED
        print("BriiBrazzzFuaazz... I'm singing a machine song.")
        pass

    def sing_variation_of_last_heard(self, var_amount=1):
        print(self.name + ": ZiiiSch... I'm singing a variation of the last song.")
        if self.last_song_variation is not None:
            self.sing(self.last_song_variation)
        else:
            print(self.model_name + ": Upps, i didn't think about a variation.")

    def quiet_please(self):
        while time.time() < self.start_time + self.current_song_duration + 1:
            pass
        print(self.name + ": I'm done singing my song. Did you like it?")
        self.currently_singing = False

    """
    ========
     Listen
    ========
    Here we want the model to prepare a new song, based on the song it just received.
    """

    def listen(self):
        print(self.name + ": Ohhuu... I'm listening to nice machine music...")
        self.last_song_imitation, spectos = Snippets.pcm_to_pcm(model=self.snippet_autoencoder,
                                                                data=self.last_song_heard,
                                                                min_size_fraction=self.min_size_fraction,
                                                                hop_length=self.hop_length,
                                                                n_fft=self.n_fft,
                                                                win_length=self.win_length,
                                                                var_amount=0)

        if spectos is not None:
            self.remembered_spectos.extend(spectos)
        else:
            print("No spectos... No vocals?")
        return spectos

    def think_about_variation(self, var_amount=1):
        print(self.name + ": Mhhh... Interesting Song, let me think about a variation of that...")
        self.last_song_variation, spectos = Snippets.pcm_to_pcm(model=self.snippet_autoencoder,
                                                                data=self.last_song_heard,
                                                                min_size_fraction=self.min_size_fraction,
                                                                hop_length=self.hop_length,
                                                                n_fft=self.n_fft,
                                                                win_length=self.win_length,
                                                                var_amount=var_amount)
        if spectos is not None:
            self.remembered_spectos.extend(spectos)
        else:
            print("No spectos... No vocals?")
        return spectos

    #     def pcm_to_pcm(self, pcm_data, variation=0):
    #         spectos = self.pcm_to_spectos(pcm_data, variation=variation)
    #         new_pcm, _ = Snippets.specto_to_pcm(model=self.snippet_autoencoder,
    #                                          data=spectos,
    #                                          hop_length=self.hop_length,
    #                                          n_fft=self.n_fft,
    #                                          win_length=self.win_length)
    #         return new_pcm

    #     def pcm_to_spectos(self, pcm_data):
    #         snippet_generator = Snippets(file_path=None,
    #                                           min_size_fraction=self.min_size_fraction,
    #                                           win_length=self.win_length,
    #                                           n_fft=self.n_fft,
    #                                           hop_length=self.hop_length)
    #         snippet_generator.data = pcm_data
    #         spectos = snippet_generator.get_snippet_spectos(delete_silence=False)

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

