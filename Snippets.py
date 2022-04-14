import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
import librosa.display
import librosa.feature.inverse
import soundfile
from IPython.display import display, Audio
import math
import os

class Snippets:
    def __init__(self, file_path, min_size_fraction, win_length, n_fft, hop_length):
        self.data, self.sr = librosa.load(file_path, sr=None, mono=True)
        self.min_size = self.sr / min_size_fraction
        self.split_points = []
        self.win_length = win_length
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _delete_silence(self, pcm, min_size):
        new_pcm = []
        for i in range(0, len(pcm), min_size):
            frame = pcm[i:i + min_size]
            rms = math.sqrt(sum(frame ** 2)) / len(frame)
            if rms > 0.00001:
                new_pcm.append(frame)
        return np.array(new_pcm).flatten()

    def get_snippet_spectos(self):
        self.data = self._delete_silence(self.data, 2 * self.sr)
        if len(self.data) > 0:
            self._get_splitPoints()
            snippets = self._generate_snippets()
            spectos = self._get_mel_spectos(snippets)
            normalized_spectos = self._normalise(np.asarray(spectos), 0, 1, -100, 100)
            return normalized_spectos
        else:
            return None

    def _get_splitPoints(self):
        for i in range(int(len(self.data) / self.min_size)):
            self.split_points.append(int((i + 1) * self.min_size))
        self.split_points = np.asarray(self.split_points)

    def _generate_snippets(self):
        snippets = []
        start = 0

        for split_point in self.split_points:
            end = split_point
            snippet = self.data[start:end]
            start = end
            if type(snippet) is np.ndarray:
                snippets.append(snippet)

        last_snippet = self.data[start:]
        last_snippet = self._apply_padding(last_snippet, self.min_size)
        if type(last_snippet) is np.ndarray:
            snippets.append(last_snippet)
        return snippets

    def _apply_padding(self, snippet, max_length):
        missing_vals = max_length - snippet.shape[0]
        if missing_vals > 0:
            snippet = np.pad(snippet, (0, int(missing_vals)), 'constant', constant_values=(0, 0))
        return snippet

    def _get_mel_spectos(self, snippets):
        spectograms = []
        for snippet in snippets:
            mel = librosa.feature.melspectrogram(y=snippet, sr=self.sr, hop_length=self.hop_length,
                                                 win_length=self.win_length, n_fft=self.n_fft, n_mels=128, fmax=16000)
            spectogram = librosa.power_to_db(mel)
            spectogram = spectogram[..., np.newaxis]
            spectograms.append(spectogram)
        return spectograms

    @staticmethod
    def _normalise(array, new_min, new_max, old_min, old_max):
        norm_array = (array - old_min) / (old_max - old_min)
        norm_array = norm_array * (new_max - new_min) + new_min
        return norm_array

    def plot_snippets(self,  plot_range=[0, 44100*4]):
        data_range = self.data[plot_range[0]:plot_range[1]]
        plt.figure(figsize=(15, 5))
        plt.plot(data_range)
        plt.hlines(0.0, plot_range[0], plot_range[1], color='r')
        plt.vlines(
            self.split_points[np.argwhere((self.split_points > plot_range[0]) & (self.split_points < plot_range[1]))],
            data_range.max(), data_range.min(), color='g')
        plt.xlabel("samples")
        plt.ylabel("amplitude")
        plt.title("Wavesets")
        plt.show();



    """
    --------------
    RECONSTRUCTION
    --------------
    """
    @classmethod
    def specto_to_pca(cls, model, data,hop_length, n_fft, win_length):
        latent_representation = model.encoder.predict(data)
        reconstructed_pca, reconstructed_specto = cls.latent_representation_to_pca(latent_representation=latent_representation,
                                                                                   model=model,
                                                                                   hop_length=hop_length,
                                                                                   n_fft=n_fft,
                                                                                   win_length=win_length)
        print(reconstructed_pca.shape)
        return reconstructed_pca, reconstructed_specto

    @classmethod
    def latent_representation_to_pca(cls, latent_representations, model, hop_length, n_fft, win_length):
        reconstructed_specto = model.decoder.predict(latent_representations)
        reconstructed_pca,_ = cls.reconstructed_spectos_to_pca(reconstructed_specto,hop_length, n_fft, win_length )
        return reconstructed_pca, reconstructed_specto

    @classmethod
    def reconstructed_spectos_to_pca(cls, spectos, hop_length, n_fft, win_length):
        reconstructed_data = np.array(0)
        clipped_spectos = []
        for i, specto in enumerate(spectos):
            reshaped_specto = specto[:, :, 0]
            denorm_specto = cls._denormalise(reshaped_specto, 0, 1, -100, 100)
            clipped_specto = np.clip(denorm_specto, -100, 20)
            #print(clipped_specto.max())
            lin_specto = librosa.db_to_power(clipped_specto)
            pca = librosa.feature.inverse.mel_to_audio(lin_specto, sr=44100,
                                                       hop_length=hop_length,
                                                       n_fft=n_fft,
                                                       win_length=win_length,
                                                       fmax = 16000)
            signal = cls._cut_zero_crossings(pca)
            reconstructed_data = np.append(reconstructed_data,signal)
            clipped_spectos.append(clipped_specto)
            print(f"Reconstructed {i + 1} of {len(spectos)} spectos", end="\r")
        reconstructed_data = reconstructed_data.flatten()
        return reconstructed_data, clipped_spectos

    @staticmethod
    def _denormalise(array, current_min, current_max, original_min, original_max):
        denorm_array = (array - current_min) / (current_max - current_min)
        denorm_array = denorm_array * (original_max - original_min) + original_min
        return denorm_array

    @staticmethod
    def _cut_zero_crossings(pca):
        zero_crossings = np.argwhere(
            (np.sign(pca[:-1]) == -1) & (np.sign(pca[1:]) == 1)
        )
        if(zero_crossings.shape[0] > 0):
            last_zero = zero_crossings[-1][0]
            first_zero = zero_crossings[0][0]
            cutted_pca = pca[first_zero:last_zero]
            return cutted_pca
        else: return pca

    @classmethod
    def plot_specto(cls, specto, name, hop_length):
        print("\n" + name + "\n")
        librosa.display.specshow(
            specto[0].reshape(specto[0].shape[0], specto[0].shape[1]),
            x_axis='time',
            y_axis='mel',
            sr=44100,
            fmax=16000,
            hop_length=hop_length,
        )
        plt.colorbar()
        plt.show()
