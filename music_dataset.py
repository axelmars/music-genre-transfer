import os

from pydub import AudioSegment
import numpy as np
import pandas as pd
import tensorflow as tf

import librosa
import librosa.display
import ast
import pickle
import re
# import mirdata.medley_solos_db
from imageio import imwrite
from pathlib import Path
from pydub.exceptions import CouldntDecodeError
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, IncrementalPCA
from hdbscan import HDBSCAN
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras import backend as K
from keras.models import Model, load_model
from keras.applications import vgg16
from keras.layers import Layer, Input, Flatten, Lambda
# from keras.preprocessing import image
from config import default_config


TRACK_ID_COL_NAME = 'Unnamed: 0'
ALL_GENRES_COL_NAME = 'track.9'
TRACKS_DIR_NAME = 'fma_small'
GENRES_COL_NAME = 'track.8'
CLASS_1_NAME = 'folk'
CLASS_2_NAME = 'rock'
CLASS_1_ID = 17
CLASS_2_ID = 12
MP3_PATH = 'C:\\Users\\Avi\\Desktop\\Uni\\ResearchProjectLab\\dataset_fma\\fma_medium'
MP3_PATH_SOLOS = 'C:\\Users\\Avi\\Desktop\\Uni\\ResearchProjectLab\\dataset_solos\\solos'
TRACKS_METADATA_FMA = 'C:/Users/Avi/Desktop/Uni/ResearchProjectLab/fma_metadata01/tracks.csv'
FEATURES_FMA = 'C:/Users/Avi/Desktop/Uni/ResearchProjectLab/fma_metadata01/features.csv'
SPECS_OUTPUT_DIR = 'C:\\Users\\Avi\\Desktop\\Uni\\ResearchProjectLab\\dataset_fma\\fma_medium_specs_img_c'
OVERLAP_SPECS_OUTPUT_DIR = f'C:\\Users\\Avi\\Desktop\\Uni\\ResearchProjectLab\\dataset_fma\\fma_medium_specs_overlap-{CLASS_1_ID}-{CLASS_2_ID}-e'
MEDLEY_SPECS_OUTPUT_DIR = f'C:\\Users\\Avi\\Desktop\\Uni\\ResearchProjectLab\\dataset_solos\\solos_specs'
WAV_OUTPUT_DIR = f'C:\\Users\\Avi\\Desktop\\Uni\\ResearchProjectLab\\dataset_fma\\fma_medium_wav-{CLASS_1_ID}-{CLASS_2_ID}'
SOLOS_DATASET_DIR = f'C:\\Users\\Avi\\Desktop\\Uni\\ResearchProjectLab\\dataset_solos\\solos'
TRACKS_LIST = 'track_paths-solos.pkl'
SORTS_LIST = 'sorts-solos.pkl'
GENRES_LIST = 'genres-solos.pkl'
TRACK_IDS_LIST = 'track_ids-solos.pkl'
SPEC_PATHS_LIST_SOLOS = 'spec_paths-solos.pkl'
TRAIN_GENRES = 'train-genres.pkl'
TRAIN_TRACK_IDS = 'train-track-ids.pkl'


def safe_parse(x):
    try:
        return ast.literal_eval(x)
    except (SyntaxError, ValueError):
        print('Warning: found malformed data.')
        return np.nan


# get set of genres
# get track id's for all songs conforming to the set of genres.

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def get_tracks_ids():
    # genres_data = pd.read_csv('music_metadata/genres.csv')
    tracks_data = pd.read_csv('music_metadata/tracks.csv').iloc[2:][[TRACK_ID_COL_NAME, ALL_GENRES_COL_NAME, GENRES_COL_NAME]]
    # genres_data = pd.read_csv('music_metadata/genres.csv')
    features_data = pd.read_csv('music_metadata/features.csv')
    print(tracks_data)
    # print(genres_data)
    print(tracks_data.columns.get_level_values(0))
    print(tracks_data[ALL_GENRES_COL_NAME][2])
    print(tracks_data[TRACK_ID_COL_NAME][2])
    # print(tracks_data)
    # tracks_data = tracks_data['track, album']
    # print(type(tracks_data.at[2, ALL_GENRES_COL_NAME]))
    # tracks_data[ALL_GENRES_COL_NAME] = pd.eval(tracks_data[ALL_GENRES_COL_NAME])
    tracks_data[ALL_GENRES_COL_NAME] = tracks_data[ALL_GENRES_COL_NAME].apply(safe_parse)
    tracks_data[GENRES_COL_NAME] = tracks_data[GENRES_COL_NAME].apply(safe_parse)
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


def list_tracks():
    track_paths = []
    genre_ids = []
    tracks_data = pd.read_csv(TRACKS_METADATA_FMA).iloc[2:][[TRACK_ID_COL_NAME, GENRES_COL_NAME]]
    tracks_data[GENRES_COL_NAME] = tracks_data[GENRES_COL_NAME].apply(safe_parse)
    tracks_data[TRACK_ID_COL_NAME] = tracks_data[TRACK_ID_COL_NAME].astype(int)
    mlb = MultiLabelBinarizer()
    tracks_data = tracks_data.join(pd.DataFrame(mlb.fit_transform(tracks_data.pop(GENRES_COL_NAME)), columns=mlb.classes_, index=tracks_data.index))
    # regex = re.compile('Rafd(\d+)_(\d+)_(\w+)_(\w+)_(\w+)_(\w+).jpg')
    base_dir = MP3_PATH  # '\\' + track_id[:3] + '\\' + track_id + '.mp3'
    track_ids = []
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        for file_name in os.listdir(folder_path):
            track_id = int(file_name[:6])
            # print(track_id)
            # print(tracks_data[tracks_data[TRACK_ID_COL_NAME] == track_id])
            # print(tracks_data[tracks_data[TRACK_ID_COL_NAME] == track_id][10] == 1)
            if (tracks_data[tracks_data[TRACK_ID_COL_NAME] == track_id][CLASS_1_ID] == 1).bool() and (tracks_data[tracks_data[TRACK_ID_COL_NAME] == track_id][CLASS_2_ID] == 0).bool():  # if the
                # genre is soundtrack and not pop
                track_paths.append(os.path.join(folder_path, file_name))
                genre_ids.append(CLASS_1_ID)
                track_ids.append(track_id)
                # print('appending ', CLASS_1_ID)
            elif (tracks_data[tracks_data[TRACK_ID_COL_NAME] == track_id][CLASS_2_ID] == 1).bool() and (tracks_data[tracks_data[TRACK_ID_COL_NAME] == track_id][CLASS_1_ID] == 0).bool():  # if the
                # genre is pop and bot soundtrack
                track_paths.append(os.path.join(folder_path, file_name))
                genre_ids.append(CLASS_2_ID)
                track_ids.append(track_id)
            # print('appending 5', CLASS_2_ID)
    print(f'num {CLASS_1_ID}: ', str(np.count_nonzero(np.array(genre_ids) == CLASS_1_ID)))
    print(f'num {CLASS_2_ID}: ', str(np.count_nonzero(np.array(genre_ids) == CLASS_2_ID)))
            # if
        # with open(self.__identity_map_path, 'r') as fd:
        #     lines = fd.read().splitlines()
        #
        # img_paths = []
        # identity_ids = []
        #
        # for line in lines:
        #     img_name, identity_id = line.split(' ')
        #     img_path = os.path.join(self.__imgs_dir, os.path.splitext(img_name)[0] + '.png')
        #
        #     img_paths.append(img_path)
        #     identity_ids.append(identity_id)
        #
        # return img_paths, identity_ids
    return track_paths, genre_ids, track_ids


def list_tracks_medley():
    track_paths = []
    genre_ids = []
    base_dir = Path(MP3_PATH_SOLOS)  # '\\' + track_id[:3] + '\\' + track_id + '.mp3'
    track_ids = []
    sorts = []
    for file_name in os.listdir(base_dir):
        if file_name[:2] != '._':
            # print(file_name)
            reg = re.search(r'Medley-solos-DB_(\w+)-(\d)_(\w{8}-\w{4}-\w{4}-\w{4}-\w{12}).wav', file_name)
            sort = reg.group(1)
            genre = reg.group(2)
            track_id = reg.group(3)
            # print(sort, genre, track_id)
            # if (tracks_data[tracks_data[TRACK_ID_COL_NAME] == track_id][CLASS_1_ID] == 1).bool() and (tracks_data[tracks_data[TRACK_ID_COL_NAME] == track_id][CLASS_2_ID] == 0).bool():  # if the
            #     # genre is soundtrack and not pop
            #     track_paths.append(os.path.join(folder_path, file_name))
            sorts.append(sort)
            genre_ids.append(genre)
            track_ids.append(track_id)
            track_paths.append(os.path.join(base_dir, file_name))
            # print('appending ', CLASS_1_ID)
        # print('appending 5', CLASS_2_ID)
    with open(TRACKS_LIST, 'wb') as f2:
        pickle.dump(track_paths, f2)

    with open(GENRES_LIST, 'wb') as f2:
        pickle.dump(genre_ids, f2)

    with open(SORTS_LIST, 'wb') as f2:
        pickle.dump(sorts, f2)

    with open(TRACK_IDS_LIST, 'wb') as f2:
        pickle.dump(track_ids, f2)

    # print(f'num {CLASS_1_ID}: ', str(np.count_nonzero(np.array(genre_ids) == CLASS_1_ID)))
    # print(f'num {CLASS_2_ID}: ', str(np.count_nonzero(np.array(genre_ids) == CLASS_2_ID)))
    return track_paths, genre_ids, track_ids, sorts


def create_spectrograms_medley(include_phase=True, double_phase=True, width=128):
    """
    loads tracks according to given ids and saves their spectrograms.
    :param tracks_ids:
    :return:
    """
    with open(TRACKS_LIST, 'rb') as f1:
        wav_file_paths = pickle.load(f1)

    with open(GENRES_LIST, 'rb') as f2:
        genre_ids = pickle.load(f2)

    with open(SORTS_LIST, 'rb') as f2:
        sorts = pickle.load(f2)

    with open(TRACK_IDS_LIST, 'rb') as f2:
        track_ids = pickle.load(f2)

    # print('num pop: ', np.count_nonzero(np.array(genre_ids) == CLASS_2_ID), ' num classical: ', np.count_nonzero(np.array(genre_ids) == CLASS_1_ID))
    spec_paths = []
    for sort in np.unique(sorts):
        Path(MEDLEY_SPECS_OUTPUT_DIR, sort).mkdir(parents=True, exist_ok=True)
    for track_path, genre_id, sort, track_id in zip(wav_file_paths, genre_ids, sorts, track_ids):
        # sample_rate, audio_data = wavfile.read(track_path)
        audio_data, sample_rate = librosa.load(track_path, sr=22050)
        # audio_data = audio_data.astype(float)
        # if len(audio_data.shape) > 1:
        #     audio_data = audio_data.mean(axis=1)
        # mel_specto = librosa.stft(y=audio_data)
        if include_phase:
            spectro = librosa.stft(y=audio_data)
            if double_phase:
                phase = np.angle(spectro)
                fbank = librosa.filters.mel(sample_rate, n_fft=2048, n_mels=128)
                phase_sin_spectro = np.dot(fbank, np.sin(phase))
                phase_cos_spectro = np.dot(fbank, np.cos(phase))
                amplitude_spectro = np.dot(fbank, np.abs(spectro) ** 2)
                amplitude_spectro = librosa.power_to_db(amplitude_spectro)
                S_dB = np.zeros(shape=(128, amplitude_spectro.shape[1], 3), dtype=np.float32)
                S_dB[:, :, 0] = amplitude_spectro
                S_dB[:, :, 1] = phase_cos_spectro
                S_dB[:, :, 2] = phase_sin_spectro
            else:
                phase_spectro = melspectrogram(S=np.angle(spectro), sr=sample_rate, is_angle=True)
                amplitude_spectro = melspectrogram(S=np.abs(spectro) ** 2, sr=sample_rate, is_angle=False)
                amplitude_spectro = librosa.power_to_db(amplitude_spectro)
                S_dB = np.zeros(shape=(128, amplitude_spectro.shape[1], 2), dtype=np.float32)
                S_dB[:, :, 0] = amplitude_spectro
                S_dB[:, :, 1] = phase_spectro
            print('ampl max: ', np.max(S_dB[:, :, 0]), 'ampl min: ', np.min(S_dB[:, :, 0]))
            print('phase cos max: ', np.max(S_dB[:, :, 1]), ' min: ', np.min(S_dB[:, :, 1]))
            print('phase sin max: ', np.max(S_dB[:, :, 2]), ' min: ', np.min(S_dB[:, :, 2]))
            print(S_dB.shape)
        else:
            mel_spectro = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
            S_dB = librosa.power_to_db(mel_spectro)
            print('max: ', np.max(S_dB), ' min: ', np.min(S_dB))

        partition = S_dB[:, :width, :]
        im_save_path = os.path.join(MEDLEY_SPECS_OUTPUT_DIR, sort, track_id + '.npy')
        with open(im_save_path, 'wb') as f:
            np.save(f, partition)                # imwrite(im_save_path, partition)
        spec_paths.append(im_save_path)

    print('num spectrograms: ', len(spec_paths))

    with open(SPEC_PATHS_LIST_SOLOS, 'wb') as f1:
        pickle.dump(spec_paths, f1)


def create_genres_only_solos():
    with open(SORTS_LIST, 'rb') as f1:
        sorts_list = pickle.load(f1)

    with open(GENRES_LIST, 'rb') as f1:
        genres_list = pickle.load(f1)

    with open(TRACK_IDS_LIST, 'rb') as f1:
        track_ids_list = pickle.load(f1)

    assert len(sorts_list) == len(genres_list) == len(track_ids_list)

    train_genres = []
    train_ids = []
    for sort, genre, track_id in zip(sorts_list, genres_list, track_ids_list):
        if sort == 'test':
            train_genres.append(genre)
            train_ids.append(track_id)

    with open(TRAIN_GENRES, 'wb') as f2:
        pickle.dump(train_genres, f2)

    with open(TRAIN_TRACK_IDS, 'wb') as f2:
        pickle.dump(train_ids, f2)


def create_spectrograms(overlap=False, include_phase=True, double_phase=True, width=128, genres_suffix='c'):
    """
    loads tracks according to given ids and saves their spectrograms.
    :param tracks_ids:
    :return:
    """
    with open(f'wav_file_paths-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'rb') as f1:
        wav_file_paths = pickle.load(f1)

    with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}-{genres_suffix}.pkl', 'rb') as f2:
        genre_ids = pickle.load(f2)

    # print('num pop: ', np.count_nonzero(np.array(genre_ids) == CLASS_2_ID), ' num classical: ', np.count_nonzero(np.array(genre_ids) == CLASS_1_ID))
    spec_paths = []
    split_genres_ids = []
    for track_path, genre_id in zip(wav_file_paths, genre_ids):
        # sample_rate, audio_data = wavfile.read(track_path)
        audio_data, sample_rate = librosa.load(track_path, sr=22050)
        # audio_data = audio_data.astype(float)
        # if len(audio_data.shape) > 1:
        #     audio_data = audio_data.mean(axis=1)
        # mel_specto = librosa.stft(y=audio_data)
        if include_phase:
            spectro = librosa.stft(y=audio_data)
            if double_phase:
                phase = np.angle(spectro)
                fbank = librosa.filters.mel(sample_rate, n_fft=2048, n_mels=128)
                phase_sin_spectro = np.dot(fbank, (np.sin(phase) + 1) * 100)
                phase_cos_spectro = np.dot(fbank, (np.cos(phase) + 1) * 100)
                print(np.abs(spectro).max(), np.abs(spectro).min())
                amplitude_spectro = np.dot(fbank, np.abs(spectro) ** 2)
                amplitude_spectro = librosa.power_to_db(amplitude_spectro)
                S_dB = np.zeros(shape=(128, amplitude_spectro.shape[1], 3), dtype=np.float32)
                S_dB[:, :, 0] = amplitude_spectro
                S_dB[:, :, 1] = phase_cos_spectro
                S_dB[:, :, 2] = phase_sin_spectro
            else:
                phase_spectro = melspectrogram(S=np.angle(spectro), sr=sample_rate, is_angle=True)
                amplitude_spectro = melspectrogram(S=np.abs(spectro) ** 2, sr=sample_rate, is_angle=False)
                amplitude_spectro = librosa.power_to_db(amplitude_spectro)
                S_dB = np.zeros(shape=(128, amplitude_spectro.shape[1], 2), dtype=np.float32)
                S_dB[:, :, 0] = amplitude_spectro
                S_dB[:, :, 1] = phase_spectro
            print('ampl max: ', np.max(S_dB[:, :, 0]), 'ampl min: ', np.min(S_dB[:, :, 0]))
            print('phase cos max: ', np.max(S_dB[:, :, 1]), ' min: ', np.min(S_dB[:, :, 1]))
            print('phase sin max: ', np.max(S_dB[:, :, 2]), ' min: ', np.min(S_dB[:, :, 2]))
        else:
            mel_spectro = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
            S_dB = librosa.power_to_db(mel_spectro)
            print('max: ', np.max(S_dB), ' min: ', np.min(S_dB))
        # S_dB = ((-80.0 - S_dB) / -80.0) * 255
        # S_dB = S_dB.astype(np.uint8)
        # spec_im = S_dB.astype(np.uint8)
        # print(mel_specto.shape)
        # print(S_dB.shape)
        track_id = track_path[-10:-4]
        if not overlap:
            if include_phase:
                for i, partition in enumerate(np.split(S_dB, [x for x in range(width, S_dB.shape[-1], width)], axis=-1)):
                    if partition.shape[-1] == width:
                        im_save_path = os.path.join(SPECS_OUTPUT_DIR, track_id + str(i) + '.tif')
                        imwrite(im_save_path, partition)
                        # imwrite(im_save_path, partition)
                        spec_paths.append(im_save_path)
                        split_genres_ids.append(genre_id)
            else:
                for i, partition in enumerate(np.split(S_dB, [x for x in range(width, S_dB.shape[1], width)], axis=1)):
                    if partition.shape[1] == width:
                        im_save_path = os.path.join(SPECS_OUTPUT_DIR, track_id + str(i) + '.tif')
                        imwrite(im_save_path, partition)
                        # imwrite(im_save_path, partition)
                        spec_paths.append(im_save_path)
                        split_genres_ids.append(genre_id)
        else:
            if include_phase:
                for i, partition in enumerate([S_dB[:, j: j + width, :] for j in range(0, S_dB.shape[1] - width, width - 32)]):
                    Path(OVERLAP_SPECS_OUTPUT_DIR).mkdir(exist_ok=True)
                    if i < 10:
                        num = '0' + str(i)
                    else:
                        num = str(i)
                    im_save_path = os.path.join(OVERLAP_SPECS_OUTPUT_DIR, track_id + num + '.npy')
                    with open(im_save_path, 'wb') as f:
                        np.save(f, partition)
                    # imwrite(im_save_path, partition)
                    spec_paths.append(im_save_path)
                    split_genres_ids.append(genre_id)
                    # print(partition.shape)
            else:
                # 0.25 overlap ==> 32 pixels overlap
                for i, partition in enumerate([S_dB[:, j: j + 128] for j in range(0, S_dB.shape[1] - width, width - 32)]):
                    Path(OVERLAP_SPECS_OUTPUT_DIR).mkdir(exist_ok=True)
                    if i < 10:
                        num = '0' + str(i)
                    else:
                        num = str(i)
                    im_save_path = os.path.join(OVERLAP_SPECS_OUTPUT_DIR, track_id + num + '.tif')
                    imwrite(im_save_path, partition)
                    # imwrite(im_save_path, partition)
                    spec_paths.append(im_save_path)
                    split_genres_ids.append(genre_id)
        # npy_save_path = os.path.join('C:\\Users\\Avi\\Desktop\\Uni\\ResearchProjectLab\\dataset_fma\\fma_small_specs_npy', track_id + '.npy')
        # np.save(npy_save_path, S_dB)
    print('num spectrograms: ', len(spec_paths))

    with open(f'spec_paths-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'wb') as f1:
        pickle.dump(spec_paths, f1)

    with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'wb') as f2:
        pickle.dump(split_genres_ids, f2)


def create_genres_only():
    """
    loads tracks according to given ids and saves their spectrograms.
    :param tracks_ids:
    :return:
    """
    with open(f'wav_file_paths-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'rb') as f1:
        wavfile_paths = pickle.load(f1)

    with open(f'spec_paths-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'rb') as f1:
        spec_paths = pickle.load(f1)

    with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'rb') as f2:
        genre_ids = pickle.load(f2)

    split_genres_ids = np.zeros(len(spec_paths))
    spec_paths = np.array([x[-12:-6] for x in spec_paths])
    print(spec_paths.shape)
    for genre_id, wav_path in zip(genre_ids, wavfile_paths):
        split_genres_ids[spec_paths == wav_path[-10: -4]] = genre_id
    for genre_id in split_genres_ids:
        print(genre_id)
    with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'wb') as f2:
        pickle.dump(split_genres_ids, f2)

# def research():
#     with open(OVERLAP_SPECS_OUTPUT_DIR, 'rb') as fd:
#         spec_paths = pickle.load(fd)
#     for spec_path in spec_paths:
#         spec = imread(spec_path)
#         print()


def load_genre(genre_id):
    with open('spec_paths.pkl', 'rb') as f1:
        spec_paths = pickle.load(f1)

    with open('genre_ids.pkl', 'rb') as f2:
        genre_ids = pickle.load(f2)

    new_spec_paths = []
    for i, current_id in enumerate(genre_ids):
        if int(current_id) == genre_id:
            new_spec_paths.append(spec_paths[i])

    print(len(new_spec_paths))

    with open('spec_paths_' + str(genre_id) + '.pkl', 'wb') as f1:
        pickle.dump(new_spec_paths, f1)


def get_pc_eigenvalues():
    with open('features-for-clustering.pkl', 'rb') as f0:
        X = np.array(pickle.load(f0))

    n_samples = X.shape[0]
    pca = PCA()
    X -= np.mean(X, axis=0)
    X /= n_samples
    # X_transform = pca.fit_transform(X)
    # cov_matrix = np.dot(X.T, X) / n_samples
    # for eigenvector in pca.components_:
    #     print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    for value in pca.explained_variance_ratio_:
        print(value)


def convert_mp3_to_wav():
    # track_paths, genre_ids, track_ids = list_tracks()
    with open(f'track_paths-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'rb') as f1:
        track_paths = pickle.load(f1)

    # with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'rb') as f2:
    #     genre_ids = pickle.load(f2)

    wav_file_paths = []
    # AudioSegment.ffmpeg = os.getcwd() + "\\ffmpeg\\bin\\ffmpeg.exe"
    AudioSegment.converter = r"C:\Users\Avi\anaconda3\envs\music-genre-transfer\Library\bin\ffmpeg.exe"
    wav_base_path = WAV_OUTPUT_DIR
    try:
        Path(WAV_OUTPUT_DIR).mkdir(parents=True)
    except FileExistsError:
        pass

    for track_path in track_paths:
        try:
            mp3_track = AudioSegment.from_mp3(track_path)  # is it mono?
            wav_file_path = os.path.join(wav_base_path, track_path[-10:-4] + '.wav')
            mp3_track.export(wav_file_path, format='wav')  # save wav file to separate directory
            wav_file_paths.append(wav_file_path)
        except CouldntDecodeError:
            print('Couldn\'t decode ', track_path)

    with open(f'wav_file_paths-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'wb') as f1:
        pickle.dump(wav_file_paths, f1)
    # with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'wb') as f2:
    #     pickle.dump(genre_ids, f2)


def reset_genres(suffix='c'):
    track_paths, genre_ids, track_ids = list_tracks()
    with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}-{suffix}.pkl', 'wb') as f2:
        pickle.dump(genre_ids, f2)


def clustering_processing(batch, pca, kmeans):
    reduced_batch = pca.fit_transform(np.array(batch, dtype=np.float32))
    print('num pca components:', pca.n_components_, 'pca explained_variance_ratio:', pca.explained_variance_ratio_)
    kmeans.partial_fit(np.array(reduced_batch, dtype=np.float32))


def create_clustered_subgenres(vgg_features=True):
    track_paths, genre_ids, track_ids = list_tracks()
    if vgg_features:
        vgg = vgg16.VGG16(include_top=False, input_shape=(128, 128, 3))
        feature_extractor = Model(inputs=vgg.input, outputs=vgg.layers[1].output)

        img = Input(shape=(128, 128, 1))
        # channel_to_add = tf.zeros((128, 128, 1), dtype=tf.float32)
        x = Lambda(lambda t: tf.tile(t, multiples=(1, 1, 1, 3)))(img)
        # x = Lambda(lambda t: tf.concat((t, [channel_to_add]), axis=-1))(img)
        x = VggNormalization()(x)
        features = feature_extractor(x)

        model = Model(inputs=img, outputs=features, name='vgg')

        print('vgg arch:')
        model.summary()
        batch_1 = []
        batch_2 = []
        count_class_1 = 0
        count_class_2 = 0
        kmeans_1 = MiniBatchKMeans(n_clusters=3)
        kmeans_2 = MiniBatchKMeans(n_clusters=4)
        pca_1 = PCA(n_components=8)
        pca_2 = PCA(n_components=6)
        for track_path, genre_id in zip(track_paths, genre_ids):
            spec_path = Path(OVERLAP_SPECS_OUTPUT_DIR, track_path[-10: -4] + '00.npy')
            try:
                img = K.constant((np.expand_dims(np.load(spec_path)[:, :, 0], axis=(0, -1))), tf.float32)
            except FileNotFoundError:
                print(track_path[-10:-4], ' not found')
                continue
            img_features = model(img)
            img_features = np.array(K.eval(K.flatten(img_features)), dtype=np.float32)
            if genre_id == CLASS_1_ID:
                count_class_1 += 1
                batch_1.append(img_features)
                if count_class_1 % 305 == 0:
                    print('performing partial fit for genre 1...')
                    # kmeans_1.partial_fit(np.array(batch_1, dtype=np.float32))
                    clustering_processing(batch_1, pca_1, kmeans_1)
                    batch_1.clear()
            if genre_id == CLASS_2_ID:
                count_class_2 += 1
                batch_2.append(img_features)
                if count_class_2 % 305 == 0:
                    print('performing partial fit for genre 2...')
                    # kmeans_2.partial_fit(np.array(batch_2, dtype=np.float32))
                    clustering_processing(batch_2, pca_2, kmeans_2)
                    batch_2.clear()
        print('performing final partial fit for genre 1...')
        clustering_processing(batch_1, pca_1, kmeans_1)
        print('performing final partial fit for genre 2...')
        clustering_processing(batch_2, pca_2, kmeans_2)
        # inference
        print('inferring clusters for spectrograms:')
        clustering_list = []
        for track_path, genre_id in zip(track_paths, genre_ids):
            spec_path = Path(OVERLAP_SPECS_OUTPUT_DIR, track_path[-10: -4] + '00.npy')
            try:
                img = K.constant((np.expand_dims(np.load(spec_path)[:, :, 0], axis=(0, -1))), dtype=tf.float32)
            except FileNotFoundError:
                continue
            img_features = model(img)
            img_features = np.array(np.expand_dims(K.eval(K.flatten(img_features)), axis=0), dtype=np.float32)
            if genre_id == CLASS_1_ID:
                img_features = pca_1.transform(np.array(img_features, dtype=np.float32))
                clustering_list.append(kmeans_1.predict(img_features)[0])
            elif genre_id == CLASS_2_ID:
                img_features = pca_2.transform(np.array(img_features, dtype=np.float32))
                clustering_list.append(kmeans_2.predict(img_features)[0] + 4)

        with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'wb') as f2:
            pickle.dump(clustering_list, f2)

    # else:
    #     features_data = pd.read_csv(FEATURES_FMA)
    #     print(features_data)
    #     print('tracks size: ', len(track_ids))
    #     # print(features_data['feature'] == 139)
    #     # for class_ids in track_ids:
    #     df_features = features_data.loc[features_data['feature'].isin(track_ids)]
    #     print(df_features)
    #     existing_track_ids = df_features['feature']
    #     existing_track_ids_idx = np.isin(np.array(track_ids), existing_track_ids)
    #     # track_ids = track_ids[existing_track_ids_idx]
    #     track_paths = np.array(track_paths)[existing_track_ids_idx]
    #     genre_ids = np.array(genre_ids)[existing_track_ids_idx]
    #     df_features = df_features.iloc[:, 1:]
    #
    #     print(df_features)
    #
    # with open('features-for-clustering.pkl', 'wb') as f2:
    #     pickle.dump(df_features, f2)
    #
    # print('features size: ', df_features.shape)
    # # print(features['feature'])
    # # labels = HDBSCAN(min_cluster_size=20).fit_predict(X=features)
    # # labels = OPTICS().fit_predict(X=features)
    # #
    # with open('clustering-genres.pkl', 'wb') as f0:
    #     pickle.dump(genre_ids, f0)

    # with open(f'track_paths-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'wb') as f2:
    #     pickle.dump(track_paths, f2)


def visualise_reduction():
    with open('features-for-clustering.pkl', 'rb') as f0:
        features = np.array(pickle.load(f0))

    with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'rb') as f2:
        genre_ids = pickle.load(f2)

    pca_res = PCA(n_components=50).fit_transform(features)
    tfse_res = TSNE(n_components=2).fit_transform(pca_res)
    # kcolors = ['red' if (genre == CLASS_1_ID) else 'blue' for genre in genre_ids]

    plt.scatter(tfse_res[:, 0], tfse_res[:, 1], c=genre_ids, cmap='rainbow', s=2, alpha=0.5)
    plt.colorbar()
    plt.show()


def cluster():
    with open('features-for-clustering.pkl', 'rb') as f0:
        features = np.array(pickle.load(f0))

    with open('clustering-genres.pkl', 'rb') as f0:
        genre_ids = np.array(pickle.load(f0))

    genre_ids = np.array(genre_ids)
    clustered_genres_ids = np.zeros(genre_ids.shape)
    genre_1_idx = genre_ids == CLASS_1_ID
    # n_samples = features.shape[0]
    # features -= np.mean(features, axis=0)
    # features /= n_samples
    # pca_res_1 = PCA(n_components=2).fit_transform(features[genre_1_idx])
    # pca_res_2 = PCA(n_components=2).fit_transform(features[~genre_1_idx])
    # genre_1_labels = KMeans(n_clusters=4, n_init=100).fit_predict(pca_res_1)
    # genre_2_labels = KMeans(n_clusters=6, n_init=100).fit_predict(pca_res_2)
    # genre_1_labels = DecisionTreeClassifier(max_clusters=4, n_init=100).fit(features[genre_1_idx])
    # genre_2_labels = DecisionTreeClassifier(n_clusters=6, n_init=100).fit_predict(features[~genre_1_idx])

    genre_1_labels = KMeans(n_clusters=4).fit_predict(features[genre_1_idx])
    genre_2_labels = KMeans(n_clusters=4).fit_predict(features[~genre_1_idx])
    clustered_genres_ids[genre_1_idx] = genre_1_labels
    clustered_genres_ids[~genre_1_idx] = genre_2_labels + max(genre_1_labels) + 1
    #
    # pca_res = PCA(n_components=50).fit_transform(features)
    # tsne_res = TSNE().fit_transform(pca_res)
    # print('applying clustering...')
    # labels = HDBSCAN(min_cluster_size=10).fit_predict(X=tsne_res)
    #
    # with open('clustering-labels.pkl', 'wb') as f1:
    #     pickle.dump(labels, f1)
    with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'wb') as f2:
        pickle.dump(clustered_genres_ids, f2)


def finetune_clustering():
    with open('features-for-clustering.pkl', 'rb') as f0:
        features = np.array(pickle.load(f0))
    n_iter = 50
    max_min_cluster_size = 50
    score = np.zeros(max_min_cluster_size - 5)
    pca_res = PCA(n_components=50).fit_transform(features)
    for i in range(5, max_min_cluster_size):
        score_iter = np.zeros(n_iter)
        print('trying with min_cluster_size', i, '...')
        for j in range(n_iter):
            print('\t iteration', j)
            print('\t\t applying TSNE...')
            tsne_iter = TSNE(n_jobs=-1).fit_transform(pca_res)
            print('\t\t applying HDBSCAN...')
            clusterer = HDBSCAN(min_cluster_size=i)
            clusterer.fit(tsne_iter)
            score_iter[j] = sum(clusterer.probabilities_ < 0.05) / features.shape[0]
        score[i-5] = np.mean(score_iter)
        print('score:', score[i - 5])
    plt.plot(np.log(score + 1))
    plt.show()
    # print('applying clustering...')
    # labels = HDBSCAN(min_cluster_size=).fit_predict(X=features)
    #
    # with open('clustering-labels.pkl', 'wb') as f1:
    #     pickle.dump(labels, f1)


def clustering_analysis():
    with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'rb') as f2:
        genre_ids = pickle.load(f2)

    # with open('clustering-labels.pkl', 'rb') as f1:
    #     labels = np.array(pickle.load(f1))

    # print(f'num {CLASS_1_ID}: ', str(np.count_nonzero(np.array(genre_ids) == CLASS_1_ID)))
    # print(f'num {CLASS_2_ID}: ', str(np.count_nonzero(np.array(genre_ids) == CLASS_2_ID)))
    # print(genre_ids)
    # print((genre_ids == CLASS_1_ID).shape)
    # print((labels == 1).shape)
    for genre in np.unique(genre_ids):
        print('genre', genre, ': ', np.count_nonzero(genre_ids == genre))

    # print(np.count_nonzero((genre_ids == CLASS_1_ID) & (labels == -1)), np.count_nonzero((genre_ids == CLASS_2_ID) & (labels == -1)))
    # for genre, label in zip(genre_ids, labels):
    #     print(genre, label)


def set_non_clustered_genres(dataset='solos'):
    if dataset == 'fma':
        track_paths, genre_ids, track_ids = list_tracks()
        with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'wb') as f2:
            pickle.dump(genre_ids, f2)
    elif dataset == 'solos':
        track_paths, genre_ids, track_ids, sorts = list_tracks_medley()
        with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}-solos.pkl', 'wb') as f2:
            pickle.dump(genre_ids, f2)
        with open(f'_ids-{CLASS_1_ID}-{CLASS_2_ID}-solos.pkl', 'wb') as f2:
            pickle.dump(genre_ids, f2)


class VggNormalization(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs * 255
        return vgg16.preprocess_input(x)


def convert_paths_to_str():
    with open(f'spec_paths-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'rb') as f3:
        spec_paths = pickle.load(f3)

    str_spec_paths = []
    for spec_path in spec_paths:
        str_spec_path = str(spec_path)
        print(str_spec_path)
        str_spec_paths.append(str_spec_path)
    with open(f'spec_paths-{CLASS_1_ID}-{CLASS_2_ID}.pkl', 'wb') as f1:
        pickle.dump(str_spec_paths, f1)


def count_genres():
    with open(f'genre_ids-{CLASS_1_ID}-{CLASS_2_ID}-128.pkl', 'rb') as f2:
        genre_ids = pickle.load(f2)
    print(np.unique(genre_ids))
    for i in np.unique(genre_ids):
        print(f'{i}: {np.count_nonzero(genre_ids == i)}')


def melspectrogram(S, sr=22050, n_filters=128, n_fft=2048, is_angle=False):
    # low_freq_mel = 0
    # high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))  # Convert Hz to Mel
    # mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)  # Equally spaced in Mel scale
    # hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    # bin = np.floor((n_fft + 1) * hz_points / sr)
    #
    # fbank = np.zeros((n_filters, int(np.floor(n_fft / 2 + 1))))
    # for m in range(1, n_filters + 1):
    #     f_m_minus = int(bin[m - 1])  # left
    #     f_m = int(bin[m])  # center
    #     f_m_plus = int(bin[m + 1])  # right
    #
    #     for k in range(f_m_minus, f_m):
    #         fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    #     for k in range(f_m, f_m_plus):
    #         fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    fbank = librosa.filters.mel(sr, n_fft, n_mels=n_filters)
    if is_angle:
        filter_banks_x = np.dot(fbank, np.cos(S))  # filter on x axis
        filter_banks_y = np.dot(fbank, np.sin(S))  # filter on y axis
        print(f'.........{np.max(filter_banks_x), np.max(filter_banks_y)}')
        filter_banks = np.arctan2(filter_banks_x, filter_banks_y)  # change back to angle in radians
    else:
        filter_banks = np.dot(fbank, S)
    return filter_banks
    # filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability

    # filter_banks = 20 * np.log10(filter_banks)  # dB


# def download_solos_dataset():
#    mirdata.medley_solos_db.download(data_home=SOLOS_DATASET_DIR, cleanup=True, force_overwrite=False)
def count_genres_solos():
    with open(f'train-genres.pkl', 'rb') as f2:
        genre_ids = pickle.load(f2)
    print(genre_ids)
    print(np.unique(genre_ids))
    for i in range(len(genre_ids)):
        genre_ids[i] = int(genre_ids[i])
    for i in np.unique(genre_ids):
        print(f'{i}: {np.count_nonzero(genre_ids == i)}')




if __name__ == '__main__':
    # create_clustered_subgenres(vgg_features=True)
    # create_genres_only()
    # clustering_analysis()
    # # finetune_clustering()
    # get_pc_eigenvalues()
    # cluster()
    # visualise_reduction()
    # create_genres_only()
    # convert_paths_to_str()
    # set_non_clustered_genres()
    # download_solos_dataset()
    # reset_genres('s')
    # list_tracks_medley()
    # create_genres_only_solos()
    # create_spectrograms(include_phase=True, double_phase=True, width=128)

    count_genres_solos()
    # set_non_clustered_genres()
    # create_spectrograms(overlap=True, include_phase=True)
    # load_genre(CLASS_2_ID)
