import librosa
from Variational_Autoencoder_alla_Valerio import VAE
import numpy as np


class SoundGenerator:
    def __init__(self, model_name, sub_folder, hop_length, n_fft, win_size):
        self.model = VAE.load(model_name)
        self.sub_folder = sub_folder
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_size = win_size

    def generate_Song(self):
        song_order = self.generate_song_order()
        spectograms = self.generate_spectograms(song_order)
        song_audio = self.convert_spectos_to_audio(spectograms)
        return song_audio

    def generate_song_order(self):
        ## Must be replaced
        path = "data/" + self.sub_folder + "song_orders.npy"
        song_orders = np.load(path, allow_pickle=True)
        print(f"Loaded {len(song_orders)} Songs. Will choose randomly")
        song_order = song_orders[np.random.randint(0, len(song_orders))]
        print(f"Song Order Shape: {song_order.shape}")
        return song_order

    def generate_spectograms(self, latent_space_coordinates):
        spectograms = self.model.decoder.predict(latent_space_coordinates)
        print(f"Spectograms Shape: {spectograms.shape}")
        return spectograms

    def _denormalise(self, array, current_min, current_max, original_min, original_max):
        denorm_array = (array - current_min) / (current_max - current_min)
        denorm_array = denorm_array * (original_max - original_min) + original_min
        return denorm_array

    def convert_spectos_to_audio(self, spectograms):
        signals = []
        for specto in spectograms:
            reshaped_specto = specto[:, :, 0]
            denorm_specto = self._denormalise(reshaped_specto, 0, 1, -100, 100)
            lin_specto = librosa.db_to_amplitude(denorm_specto)
            pca = librosa.istft(lin_specto, hop_length=self.hop_length, n_fft=self.n_fft, win_length=self.win_size)
            signal = self.cut_on_zero_edge(pca)
            signals.extend(signal)
        print(f"One Waveset has {signals[0].shape} samples")
        print(f"This song has {len(signals)} wavesets.")
        return signals

    def cut_on_zero_edge(self, pca):
        zero_crossings = np.argwhere((np.sign(pca[:-1]) == -1) & (np.sign(pca[1:]) == 1))
        last_zero = zero_crossings[-1][0]
        first_zero = zero_crossings[0][0]
        cutted_pca = pca[first_zero:last_zero]
        return cutted_pca


if __name__ == "__main__":

    WIN_LENGTH = int(44100 / 8)
    HOP_LENGTH = int(44100 / 23)
    N_FFT = int(44100 / 8)

    sound_generator = SoundGenerator("VAE_2D_1.0_24_2500samples", "1.0_24", HOP_LENGTH, N_FFT, WIN_LENGTH)
    sound_generator.generate_Song() 