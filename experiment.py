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
from imageio import imread
import librosa.display
import librosa.feature
# from keras.layers import GRU, Input, Embedding, Reshape, Conv1D, Conv2D
# from keras.models import Model
from skimage.color import rgb2gray
# import tensorflow as tf
import imageio

TRACK_ID_COL_NAME = 'Unnamed: 0'
ALL_GENRES_COL_NAME = 'track.9'
TRACKS_DIR_NAME = 'fma_small'
GENRES_COL_NAME = 'track.8'
CLASS_1_NAME = 'orchestral'
CLASS_2_NAME = 'pop'
SPECS_OUTPUT_DIR = 'C:\\Users\\Avi\\Desktop\\Uni\\ResearchProjectLab\\dataset_fma\\fma_medium_specs_img'


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
    tracks_data = pd.read_csv('music_metadata/tracks.csv').iloc[2:][[TRACK_ID_COL_NAME, ALL_GENRES_COL_NAME]]
    genres_data = pd.read_csv('music_metadata/genres.csv')
    features_data = pd.read_csv('music_metadata/features.csv')
    print(tracks_data)
    print(genres_data)
    print(tracks_data.columns.get_level_values(0))
    print(tracks_data[ALL_GENRES_COL_NAME][2])
    print(tracks_data[TRACK_ID_COL_NAME][2])
    # print(tracks_data)
    # tracks_data = tracks_data['track, album']
    print(type(tracks_data.at[2, ALL_GENRES_COL_NAME]))
    # tracks_data[ALL_GENRES_COL_NAME] = pd.eval(tracks_data[ALL_GENRES_COL_NAME])
    tracks_data[[ALL_GENRES_COL_NAME, GENRES_COL_NAME]] = tracks_data[[ALL_GENRES_COL_NAME, GENRES_COL_NAME]].apply(safe_parse)
    mlb = MultiLabelBinarizer()
    tracks_data = tracks_data.join(pd.DataFrame(mlb.fit_transform(tracks_data.pop(GENRES_COL_NAME)), columns=mlb.classes, index=tracks_data.index))
    print(tracks_data.columns)
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


def gru_experiment():
    inputs = Input(shape=(10, 8))
    output, hidden = GRU(4, return_sequences=True, return_state=True)(inputs)
    model = Model(inputs=inputs, outputs=[output, hidden])
    data = np.random.normal(size=[16, 10, 8])
    print(model.predict(data))


def embedding_experiment():
    identity = Input(shape=(1,))

    identity_embedding = Embedding(input_dim=2, output_dim=256, name='identity-embedding')(identity)
    identity_embedding = Reshape(target_shape=(256,))(identity_embedding)

    model = Model(inputs=identity, outputs=identity_embedding)

    sample_rate, audio_data = wavfile.read('waved_sound.wav')
    audio_data = audio_data.astype(float)
    if audio_data.shape[1] == 2:
        audio_data = np.round(audio_data.sum(axis=1) / 2)
    mel_specto = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    S_dB = librosa.power_to_db(mel_specto, ref=np.max)
    data = S_dB

    plt.imshow(data)
    plt.show()
    # print(data)
    print(data.shape)
    print(model.predict(data))
    #
    # print('identity embedding:')
    # model.summary()

    return model


def conv1d_experiment():
    input_shape = (4, 128, 1033)
    x = tf.random.normal(input_shape)
    y = Conv1D(128, 7, padding='same', activation='relu', input_shape=input_shape)(x)

    print(y.shape)


if __name__ == '__main__':
    # # AudioSegment.ffmpeg = os.getcwd() + "\\ffmpeg\\bin\\ffmpeg.exe"
    # # # print(AudioSegment.ffmpeg)
    # AudioSegment.converter = r"C:\Users\Avi\anaconda3\envs\music-genre-transfer\Library\bin\ffmpeg.exe"
    # # path = r'C:\Users\Avi\Desktop\Uni\ResearchProjectLab\code_samples\music-genre-transfer'
    # # os.chdir(path)
    # # dirs = os.listdir(path)
    # # for file in dirs:
    # #     print(file)
    # sound = AudioSegment.from_mp3('007713.mp3')
    # sound.export('waved_acous.wav', format='wav')
    # samples, sample_rate = librosa.load('waved_acous.wav', sr=22050)
    # print(sample_rate, samples.shape)
    # samples = samples.astype(float)
    # # audio_data = samples.sum(axis=1) / 2
    # mel_specto = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
    # S_dB = librosa.power_to_db(mel_specto, ref=np.max)
    # # print(mel_specto.shape)
    # # print(S_dB.shape)
    # save_path = 'test2.png'
    # plt.imsave(save_path, S_dB)
    # # gru_experiment()
    # # embedding_experiment()
    # # subprocess.call(['ffmpeg', '-i', '035545.mp3', 'waved_sound.wav'])
    # # get_orchestral_tracks()
    # spec = imread('test2.png')
    # print(rgb2gray(spec).shape)
    # # conv1d_experiment()
    img = imageio.imread(os.path.join(SPECS_OUTPUT_DIR, '0051785.png'))
    print(img)
    img = img.astype(np.float32)
    print(img)
    img /= 255.0
    print(img)
