import os

from pydub import AudioSegment
import numpy as np
import pandas as pd
import librosa
import librosa.display
import ast
import pickle
from imageio import imwrite, imread

from scipy.io import wavfile

from sklearn.preprocessing import MultiLabelBinarizer

TRACK_ID_COL_NAME = 'Unnamed: 0'
ALL_GENRES_COL_NAME = 'track.9'
TRACKS_DIR_NAME = 'fma_small'
GENRES_COL_NAME = 'track.8'
CLASS_1_NAME = 'instrumental'
CLASS_2_NAME = 'pop'
CLASS_1_ID = 18
CLASS_2_ID = 10
MP3_PATH = 'C:\\Users\\Avi\\Desktop\\Uni\\ResearchProjectLab\\dataset_fma\\fma_medium'
TRACKS_METADATA_FMA = 'C:/Users/Avi/Desktop/Uni/ResearchProjectLab/fma_metadata01/tracks.csv'
SPECS_OUTPUT_DIR = 'C:\\Users\\Avi\\Desktop\\Uni\\ResearchProjectLab\\dataset_fma\\fma_medium_specs_img'
WAV_OUTPUT_DIR = 'C:\\Users\\Avi\\Desktop\\Uni\\ResearchProjectLab\\dataset_fma\\fma_medium_wav'


def safe_parse(x):
    try:
        return ast.literal_eval(x)
    except (SyntaxError, ValueError):
        print('Warning: found malformed data.')
        return np.nan


# get set of genres
# get track id's for all songs conforming to the set of genres.

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
                # print('appending ', CLASS_1_ID)
            elif (tracks_data[tracks_data[TRACK_ID_COL_NAME] == track_id][CLASS_2_ID] == 1).bool() and (tracks_data[tracks_data[TRACK_ID_COL_NAME] == track_id][CLASS_1_ID] == 0).bool():  # if the
                # genre is pop and bot soundtrack
                track_paths.append(os.path.join(folder_path, file_name))
                genre_ids.append(CLASS_2_ID)
                # print('appending 5', CLASS_2_ID)
    print('num instrumental: ', str(np.count_nonzero(np.array(genre_ids) == CLASS_1_ID)))
    print('num pop: ', str(np.count_nonzero(np.array(genre_ids) == CLASS_2_ID)))
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
    return track_paths, genre_ids


def create_spectrograms():
    """
    loads tracks according to given ids and saves their spectrograms.
    :param tracks_ids:
    :return:
    """
    with open('wav_file_paths.pkl', 'rb') as f1:
        wav_file_paths = pickle.load(f1)

    with open('genre_ids.pkl', 'rb') as f2:
        genre_ids = pickle.load(f2)

    # print('num pop: ', np.count_nonzero(np.array(genre_ids) == CLASS_2_ID), ' num classical: ', np.count_nonzero(np.array(genre_ids) == CLASS_1_ID))
    spec_paths = []
    split_genres_ids = []
    for track_path, genre_id in zip(wav_file_paths, genre_ids):
        # sample_rate, audio_data = wavfile.read(track_path)
        audio_data, sample_rate = librosa.load(track_path, sr=22050)
        # audio_data = audio_data.astype(float)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        mel_specto = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        S_dB = librosa.power_to_db(mel_specto, ref=np.max)
        S_dB = ((-80.0 - S_dB) / -80.0) * 255
        S_dB = S_dB.astype(np.uint8)
        # spec_im = S_dB.astype(np.uint8)
        # print(mel_specto.shape)
        # print(S_dB.shape)
        track_id = track_path[-10:-4]
        for i, partition in enumerate(np.split(S_dB, [x for x in range(128, S_dB.shape[1], 128)], axis=1)):
            if partition.shape[1] == 128:
                im_save_path = os.path.join(SPECS_OUTPUT_DIR, track_id + str(i) + '.png')
                imwrite(im_save_path, partition)
                spec_paths.append(im_save_path)
                split_genres_ids.append(genre_id)
        # npy_save_path = os.path.join('C:\\Users\\Avi\\Desktop\\Uni\\ResearchProjectLab\\dataset_fma\\fma_small_specs_npy', track_id + '.npy')
        # np.save(npy_save_path, S_dB)
    print('num spectrograms: ', len(spec_paths))

    with open('spec_paths.pkl', 'wb') as f1:
        pickle.dump(spec_paths, f1)

    with open('genre_ids.pkl', 'wb') as f2:
        pickle.dump(split_genres_ids, f2)


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


def convert_mp3_to_wav():
    track_paths, genre_ids = list_tracks()
    wav_file_paths = []
    AudioSegment.ffmpeg = os.getcwd() + "\\ffmpeg\\bin\\ffmpeg.exe"
    AudioSegment.converter = r"C:\Users\Avi\anaconda3\envs\music-genre-transfer\Library\bin\ffmpeg.exe"
    wav_base_path = WAV_OUTPUT_DIR
    for track_path in track_paths:
        mp3_track = AudioSegment.from_mp3(track_path)  # is it mono?
        wav_file_path = os.path.join(wav_base_path, track_path[-10:-4] + '.wav')
        mp3_track.export(wav_file_path, format='wav')  # save wav file to separate directory
        wav_file_paths.append(wav_file_path)

    with open('wav_file_paths.pkl', 'wb') as f1:
        pickle.dump(wav_file_paths, f1)

    with open('genre_ids.pkl', 'wb') as f2:
        pickle.dump(genre_ids, f2)


if __name__ == '__main__':
    # convert_mp3_to_wav()
    # create_spectrograms()
    load_genre(CLASS_2_ID)
