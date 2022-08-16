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
import time


class Snippets:
    def __init__(self, file_path, min_size_fraction, win_length, n_fft, hop_length):
        if file_path is not None:
            self.data, self.sr = librosa.load(file_path, sr=None, mono=True)
        else:
            self.data = np.array(0)
            self.sr = 44100
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

    def get_snippet_spectos(self, delete_silence=True):
        if delete_silence:
            self.data = self._delete_silence(self.data, 2 * self.sr)
        if len(self.data) > self.min_size:
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

    def plot_snippets(self, plot_range=[0, 44100 * 4]):
        data_range = self.data[plot_range[0]:plot_range[1]]
        plt.figure(figsize=(15, 5))
        plt.plot(data_range)
        plt.hlines(0.0, plot_range[0], plot_range[1], color='r')
        plt.vlines(
            self.split_points[np.argwhere((self.split_points > plot_range[0]) & (self.split_points < plot_range[1]))],
            data_range.max(), data_range.min(), color='g')
        plt.xlabel("samples")
        plt.ylabel("amplitude")
        plt.title("Snippets")
        plt.show()

    """
    --------------
    RECONSTRUCTION
    --------------
    """

    @classmethod
    def pcm_to_pcm(cls, model, data, min_size_fraction, hop_length, n_fft, win_length, var_amount=0):
        spectos = cls.pcm_to_specto(pcm_data=data,
                                    min_size_fraction=min_size_fraction,
                                    hop_length=hop_length,
                                    win_length=win_length,
                                    n_fft=n_fft)

        new_pcm, _ = cls.specto_to_pcm(model=model,
                                       data=spectos,
                                       hop_length=hop_length,
                                       n_fft=n_fft,
                                       win_length=win_length,
                                       variation=var_amount)
        return new_pcm, spectos

    @classmethod
    def pcm_to_specto(cls, pcm_data, min_size_fraction, win_length, n_fft, hop_length):
        snippet_generator = Snippets(file_path=None,
                                     min_size_fraction=min_size_fraction,
                                     win_length=win_length,
                                     n_fft=n_fft,
                                     hop_length=hop_length)
        snippet_generator.data = pcm_data
        spectos = snippet_generator.get_snippet_spectos(delete_silence=False)
        return spectos

    @classmethod
    def specto_to_pcm(cls, model, data, hop_length, n_fft, win_length, variation=0):

        latent_representation = model.encoder.predict(data)
        latent_representation += np.random.normal(0, variation, latent_representation.shape)

        reconstructed_pcm, reconstructed_specto = cls.latent_representation_to_pcm(
            latent_representations=latent_representation,
            model=model,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length)
        return reconstructed_pcm, reconstructed_specto

    @classmethod
    def latent_representation_to_pcm(cls, latent_representations, model, hop_length, n_fft, win_length):
        # print("Getting latent representation of the spectos...")
        reconstructed_specto = model.decoder.predict(latent_representations)
        # print("Getting PCM from spectos...")
        reconstructed_pcm, _ = cls.reconstructed_spectos_to_pcm(spectos=reconstructed_specto,
                                                                hop_length=hop_length,
                                                                n_fft=n_fft,
                                                                win_length=win_length)
        return reconstructed_pcm, reconstructed_specto

    @classmethod
    def reconstructed_spectos_to_pcm(cls, spectos, hop_length, n_fft, win_length):
        reconstructed_data = np.array(0)
        clipped_spectos = []
        for i, specto in enumerate(spectos):
            reshaped_specto = specto[:, :, 0]
            print("\n\noriginal max: " + str(reshaped_specto.max()))
            denorm_specto = cls._denormalise(reshaped_specto, 0, 1, -100, 100)
            print("denorm max: " + str(denorm_specto.max()))
            clipped_specto = np.clip(denorm_specto, -100, 20)
            lin_specto = librosa.db_to_power(clipped_specto)
            print("Shape of Lin Specto: " + str(lin_specto.shape))
            print("max: " + str(lin_specto.max()))
            print("min: " + str(lin_specto.min()))
            print("mean: " + str(np.mean(lin_specto)))

            start_time = time.time()
            print("Start Transform")
            stft = librosa.feature.inverse.mel_to_stft(M=lin_specto,
                                                       sr=44100,
                                                       n_fft=n_fft,
                                                       fmax=16000)

            stft_time = time.time()
            print("Transformed to stft in " + str(time.time() - start_time))
            print("Shape of STFT: " + str(stft.shape))
            pcm = librosa.griffinlim(S=stft,
                                     hop_length=hop_length,
                                     win_length=win_length)
            print("Transformed to pcm in " + str(time.time() - stft_time))
            print("Shape of pcm: " + str(pcm.shape))
            signal = cls._cut_zero_crossings(pcm)

            if i > 0:
                reconstructed_data = cls._crossfade(reconstructed_data, signal, 820)
            else:
                reconstructed_data = np.append(reconstructed_data, signal)
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
    def _cut_zero_crossings(pcm):
        zero_crossings = np.argwhere(
            (np.sign(pcm[:-1]) == -1) & (np.sign(pcm[1:]) == 1)
        )
        if (zero_crossings.shape[0] > 0):
            last_zero = zero_crossings[-1][0]
            first_zero = zero_crossings[0][0]
            cutted_pcm = pcm[first_zero:last_zero]
            return cutted_pcm
        else:
            return pcm

    @staticmethod
    def _crossfade(array_a, array_b, fadeTime):
        fade_in = np.sqrt((1 + np.linspace(-1, 1, fadeTime)) * 0.5)
        fade_out = np.sqrt((1 - np.linspace(-1, 1, fadeTime)) * 0.5)
        fade_in = np.pad(fade_in, (0, array_b.shape[0] - fadeTime), 'constant', constant_values=(1, 1))
        fade_out = np.pad(fade_out, (array_a.shape[0] - fadeTime, 0), 'constant', constant_values=(1, 1))
        array_a = array_a * fade_out
        array_b = array_b * fade_in
        array_a[-fadeTime:] += array_b[:fadeTime]
        result = np.append(array_a, array_b[fadeTime:])
        return result

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
