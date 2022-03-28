import librosa
from Variational_Autoencoder_alla_Valerio import VAE
import numpy as np
import pickle

class SoundGenerator:
    def __init__(self, modelName, subfolder, hop_length):
        self.model = model = VAE.load(modelName)
        self.subfolder = subfolder
        self.hop_length = hop_length

    def generate_Song(self):
        song_order = self.generate_song_order()
        spectograms = self.generate_spectograms(song_order)
        wavesets_audio = self.convert_spectos_to_audio(spectograms)
        song_audio = self.concatenate_wavesets(wavesets_audio)
        return song_audio


    def generate_song_order(self):
        ## Must be replaced
        path = "data/" + self.subfolder + "song_orders.npy"
        song_orders = np.load(path, allow_pickle=True)
        print(f"Loaded {len(song_orders)} Songs. Will choose randomly")
        song_order = song_orders[np.random.randint(0, len(song_orders))]
        print(f"Song Order Shape: {song_order.shape}")
        return song_order

    def generate_spectograms(self, latent_space_coordinates):
        spectograms = self.model.decoder.predict(latent_space_coordinates)
        print(f"Spectograms Shape: {spectograms.shape}")
        return spectograms

    def _denormalise(self, array, original_min, original_max):
        denorm_array = (array - array.min()) / (array.max() - array.min())
        denorm_array = denorm_array * (original_max - original_min) + original_min
        return denorm_array

    def convert_spectos_to_audio(self, spectograms):
        signals = []
        for specto in spectograms:
            reshaped_specto = specto[:, :, 0]
            denorm_specto = self._denormalise(reshaped_specto, -100, 0)
            lin_specto = librosa.db_to_amplitude(denorm_specto)
            signal = librosa.istft(lin_specto, hop_length=self.hop_length)
            signals.append(signal)
        print(f"One Waveset has {signals[0].shape} samples")
        print(f"This song has {len(signals)} wavesets.")
        return signals

    def concatenate_wavesets(self, signals):
        amplitudes = np.asarray(signals).flatten()
        print(f"This song has {amplitudes.shape} samples")
        return amplitudes

if __name__ == "__main__":
    sound_generator = SoundGenerator("VAE_2D_0.5_64", "0.5_64", 360)
    sound_generator.generate_Song()