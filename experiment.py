from pydub import AudioSegment
from scipy.io import wavfile

import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import os
from imageio import imread, imwrite
import librosa.display
import librosa.feature

# from keras.layers import GRU, Input, Embedding, Reshape, Conv1D, Conv2D
# from keras.models import Model
from skimage.color import rgb2gray
# import tensorflow as tf

TRACK_ID_COL_NAME = 'Unnamed: 0'
ALL_GENRES_COL_NAME = 'track.9'
TRACKS_DIR_NAME = 'fma_small'
GENRES_COL_NAME = 'track.8'
CLASS_1_NAME = 'orchestral'
CLASS_2_NAME = 'pop'


def safe_parse(x):
    try:
        return ast.literal_eval(x)
    except (SyntaxError, ValueError):
        print('Warning: found malformed data.')
        return np.nan


def get_style(x, tracks_data, genres_data):
    track_id = x[['track_id']]



def get_tracks_ids():
    # genres_data = pd.read_csv('music_metadata/genres.csv')
    tracks_data = pd.read_csv('music_metadata/tracks.csv').iloc[2:][[TRACK_ID_COL_NAME, ALL_GENRES_COL_NAME, GENRES_COL_NAME]]
    genres_data = pd.read_csv('music_metadata/genres.csv')
    features_data = pd.read_csv('music_metadata/features.csv')
    print(tracks_data)
    print(genres_data)
    print(tracks_data.columns.get_level_values(0))
    print(tracks_data[ALL_GENRES_COL_NAME][2])
    print(tracks_data[TRACK_ID_COL_NAME][2])
    # print(tracks_data)
    # tracks_data = tracks_data['track, album']
    # print(type(tracks_data.at[2, ALL_GENRES_COL_NAME]))
    # tracks_data[ALL_GENRES_COL_NAME] = pd.eval(tracks_data[ALL_GENRES_COL_NAME])
    # tracks_data[ALL_GENRES_COL_NAME] = tracks_data[ALL_GENRES_COL_NAME].apply(safe_parse)
    tracks_data[GENRES_COL_NAME] = tracks_data[GENRES_COL_NAME].apply(safe_parse)
    mlb = MultiLabelBinarizer()
    tracks_data = tracks_data.join(pd.DataFrame(mlb.fit_transform(tracks_data.pop(GENRES_COL_NAME)), columns=mlb.classes_, index=tracks_data.index))
    print(tracks_data)
    print(np.count_nonzero(tracks_data[5]))
    print(tracks_data[tracks_data[TRACK_ID_COL_NAME] == 26583])
    # print(tracks_data[tracks_data[TRACK_ID_COL_NAME] == '5'][21])
    # orchestral genres: (symphony, soundtrack&(classical|pop|rock))
    # pop genres: (pop[not synthpop]). Make certain intersection with above is empty.
    orchestral_tracks_ids = tracks_data[tracks_data['something']]
    orchestral_tracks_ids = orchestral_tracks_ids.assign(class_name=CLASS_1_NAME)
    pop_tracks_ids = tracks_data[tracks_data['something']]
    pop_tracks_ids = pop_tracks_ids.assign(class_name=CLASS_2_NAME)
    track_ids = pd.concat([orchestral_tracks_ids, pop_tracks_ids]).drop_duplicates(subset=TRACK_ID_COL_NAME, keep='first').reset_index(drop=True)

    track_ids = track_ids.assign(style=tracks_data[GENRES_COL_NAME])
    print(track_ids)
    return track_ids


def load_tracks(tracks_ids):
    """
    loads tracks according to given ids and saves their spectrograms.
    :param tracks_ids:
    :return:
    """
    for track_id in tracks_ids:
        sample_rate, audio_data = wavfile.read('\\' + TRACKS_DIR_NAME + '\\' + track_id[:3] + '\\' + track_id + '.mp3')
        audio_data = audio_data.astype(float)
        if audio_data.shape[1] == 2:
            audio_data = audio_data.sum(axis=1) / 2
        mel_specto = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        S_dB = librosa.power_to_db(mel_specto, ref=np.max)
        # print(mel_specto.shape)
        # print(S_dB.shape)
        save_path = 'spectrograms\\' + track_id + '.png'
        plt.imsave(save_path, S_dB)


def turn_tracks_into_short_tracks():
    """
    splits tracks into short (find out how long spectrogram, how long can lord take.
    :return:
    """
    pass


def preprocess():
    tracks_ids = get_tracks_ids()
    load_tracks(tracks_ids)


# def gru_experiment():
#     inputs = Input(shape=(10, 8))
#     output, hidden = GRU(4, return_sequences=True, return_state=True)(inputs)
#     model = Model(inputs=inputs, outputs=[output, hidden])
#     data = np.random.normal(size=[16, 10, 8])
#     print(model.predict(data))


# def embedding_experiment():
#     identity = Input(shape=(1,))
#
#     identity_embedding = Embedding(input_dim=2, output_dim=256, name='identity-embedding')(identity)
#     identity_embedding = Reshape(target_shape=(256,))(identity_embedding)
#
#     model = Model(inputs=identity, outputs=identity_embedding)
#
#     sample_rate, audio_data = wavfile.read('waved_sound.wav')
#     audio_data = audio_data.astype(float)
#     if audio_data.shape[1] == 2:
#         audio_data = np.round(audio_data.sum(axis=1) / 2)
#     mel_specto = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
#     S_dB = librosa.power_to_db(mel_specto, ref=np.max)
#     data = S_dB
#
#     plt.imshow(data)
#     plt.show()
#     # print(data)
#     print(data.shape)
#     print(model.predict(data))
#     #
#     # print('identity embedding:')
#     # model.summary()
#
#     return model


# def conv1d_experiment():
#     input_shape = (4, 128, 1033)
#     x = tf.random.normal(input_shape)
#     y = Conv1D(128, 7, padding='same', activation='relu', input_shape=input_shape)(x)
#
#     print(y.shape)
#
def normalize8(I):
    mn = I.min()
    mx = I.max()

    mx -= mn

    I = ((I - mn)/mx) * 255
    return I.astype(np.uint8)


def denormalize8(I):
    I = (())

def binomial_mask(a=1, x=1, im_shape=(128, 128, 3)):
    n = int(.25 * im_shape[1]) - 1
    term = pow(a, n)
    # print(term, end=" ")
    series_list = [term]
    # Computing and printing remaining terms
    for i in range(1, n + 1):
        # Find current term using previous terms
        # We increment power of x by 1, decrement
        # power of a by 1 and compute nCi using
        # previous term by multiplying previous
        # term with (n - i + 1)/i
        term = int(term * x * (n - i + 1) / (i * a))
        series_list.append(term)
    ser_array = np.array(series_list)
    ser_array_squash = ser_array / np.max(ser_array) / 2
    half = int(n / 2)
    ser_array_squash = np.concatenate((ser_array_squash[: half], 1 - ser_array_squash[half:]))[None, :, None]
    print(ser_array_squash.shape)
    mask = np.zeros((im_shape[0], 2 * im_shape[1] - int(.25 * im_shape[1]), 3))
    mask[:, im_shape[1]: im_shape[1] + int(.75 * im_shape[1]), :] = np.ones((im_shape[0], int(.75 * im_shape[1]), 3))
    mask[:, int(.75 * im_shape[1]): im_shape[1], :] = np.tile(ser_array_squash, (im_shape[0], 1, 1))
    return mask

if __name__ == '__main__':
    # AudioSegment.ffmpeg = os.getcwd() + "\\ffmpeg\\bin\\ffmpeg.exe"
    # # print(AudioSegment.ffmpeg)
    # AudioSegment.converter = r"C:\Users\Avi\anaconda3\envs\music-genre-transfer\Library\bin\ffmpeg.exe"
    # # path = r'C:\Users\Avi\Desktop\Uni\ResearchProjectLab\code_samples\music-genre-transfer'
    # # os.chdir(path)
    # # dirs = os.listdir(path)
    # # for file in dirs:
    # #     print(file)
    # # sound = AudioSegment.from_mp3('007713.mp3')
    # # sound.export('waved_acous.wav', format='wav')
    # samples, sample_rate = librosa.load('061679.wav', sr=22050)
    #
    # # stft_specto = np.abs(librosa.stft(y=samples))
    # spectro = librosa.stft(y=samples)
    # phase_spectro = librosa.feature.melspectrogram(S=np.angle(spectro), sr=sample_rate)
    # amplitude_spectro = librosa.feature.melspectrogram(S=np.abs(spectro) ** 2, sr=sample_rate)
    # amplitude_spectro = librosa.power_to_db(amplitude_spectro)
    #
    # p_a_spetro = np.zeros((128, 128, 2), dtype=np.float32)
    #
    # print(p_a_spetro.dtype, p_a_spetro.shape)
    # print(amplitude_spectro.dtype, amplitude_spectro.shape)
    #
    # np.save('spectrogram_ampl_phase.npy', p_a_spetro)
    # np.save('spectrogram_ampl.npy', amplitude_spectro)
    # binomial_mask()
    samples, sample_rate = librosa.load('0019410-8.wav', sr=22050)
    print(max(samples))

    # out_spectro = librosa.feature.inverse.mel_to_stft(mel_phase_spectro)
    # print(out_spectro.dtype, out_spectro.shape)
    # print(np.allclose(out_spectro, phase_spectro))

    # mel_specto = librosa.feature.melspectrogram(samples, sr=sample_rate)
    #
    # print(mel_specto.shape)
    # # spectro = librosa.stft(y=samples)
    # # S_dB = mel_specto
    # S_dB = librosa.power_to_db(mel_specto)
    #
    # print(S_dB.shape)
    # print('max: ', np.max(S_dB), ' min: ', np.min(S_dB))
    # print(S_dB)
    # # S_dB = (-80.0 - S_dB) / -80
    # # S_dB = S_dB.astype(np.uint8)
    #
    # save_path = 'test2.tif'
    # print(S_dB)
    # imwrite(save_path, S_dB)
    # # np.save(save_path, spectro)
    # # spec = np.load('test2.npy')
    # spec = imread('test2.tif')
    # print(spec.dtype)
    # # spec = (spec * (-80.0) + 80.0) * -1
    # # print(spec)
    # # # spec = spec.astype(np.float32)
    # # print(spec)
    # # # spec = spec / 255.0
    # # print(spec)
    # # np.save('test2.npy', spec)
    # # spec = np.load('test2.npy')
    # print(spec)
    # # print('db_to_power done')
    # # S = librosa.feature.inverse.mel_to_stft(spec, sr=22050)
    # # print('mel_to_stft done')
    # spec = librosa.feature.inverse.db_to_power(spec)
    # audio = librosa.feature.inverse.mel_to_audio(spec)
    # # audio = librosa.istft(spec)
    # # audio = librosa.griffinlim(S)
    # print('stft_to_audio done')
    # # audio.export('reconstructed_wav.wav', format='wav')
    # # audio = np.asarray(audio, dtype=np.int16)
    # print(audio)
    #
    # wavfile.write('reconstructed_wav.wav', sample_rate, audio)
    # # print(spec.shape)
    # # print(spec)
    # plt.imshow(spec)
    # plt.show()

    # conv1d_experiment()
    # ids = get_tracks_ids()