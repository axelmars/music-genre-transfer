import os
import argparse
import numpy as np
import pickle
from imageio import imread, imwrite
import re
import librosa
from config import default_config
from scipy.io import wavfile
# from model import evaluation
from model.network import Converter
from pathlib import Path

SAMPLE_RATE = 22050
CLASS_1_ID = 17
CLASS_2_ID = 12
CLASS_1_SUB = 1
CLASS_2_SUB = 8
paddings = [[0, 0], [0, 0], [1, 0]]
IDENTITY_TRANSFORM_DIR = f'identity_transform_{CLASS_1_SUB}-{CLASS_2_SUB}-ampl'
GENRE_TRANSFORM_DIR = f'genre_transform_{CLASS_1_SUB}-{CLASS_2_SUB}-ampl'

class Inferer:
    def __init__(self, args):
        # assets = AssetManager(args.base_dir)
        model_dir = os.path.join(args.base_dir, 'cache', 'models', args.model_name)
        # model_dir = assets.recreate_model_dir(args.model_name)
        self.__base_dir = args.base_dir
        self.__dataset_dir = args.dataset_dir
        self.__num_samples = args.num_samples
        self.__model_dir = model_dir
        self.__include_encoders = args.is_test
        self.__overlap = args.is_overlapping

        self.__converter = Converter.load(model_dir, include_encoders=self.__include_encoders)
        # self.__converter.pose_encoder.compile()

    def infer(self, identity=True):
        with open(os.path.join(self.__base_dir, f'bin/genre_ids-{CLASS_1_ID}-{CLASS_2_ID}-128.pkl'), 'rb') as f2:
            genre_ids = pickle.load(f2)

        with open(os.path.join(self.__base_dir, f'bin/spec_paths-{CLASS_1_ID}-{CLASS_2_ID}-128.pkl'), 'rb') as fd:
            spec_paths = pickle.load(fd)

        genre_ids = np.array(genre_ids)
        spec_paths = np.array(spec_paths)
        genre_ids[genre_ids == CLASS_2_SUB] = CLASS_2_SUB
        print('genre_ids class 2 shape: ', np.count_nonzero(genre_ids == CLASS_2_SUB))
        genre_ids[genre_ids == CLASS_1_SUB] = CLASS_1_SUB
        if not self.__include_encoders:
            indices = np.load(os.path.join(self.__base_dir, 'bin/train_idx.npy'))
        else:
            indices = np.load(os.path.join(self.__base_dir, 'bin/test_idx.npy'))
        sample_paths = spec_paths[indices]
        sample_genres = genre_ids[indices]
        print('sample_paths shape: ', sample_paths.shape)
        sample_paths_0 = sample_paths[sample_genres == CLASS_1_SUB]
        sample_paths_1 = sample_paths[sample_genres == CLASS_2_SUB]
        print('sample_paths_1 shape: ', sample_paths_1.shape)
        sample_paths_0 = sample_paths_0[np.random.choice(sample_paths_0.shape[0], size=self.__num_samples, replace=False)]
        print('sample_paths_0 shape: ', sample_paths_0.shape)
        sample_paths_1 = sample_paths_1[np.random.choice(sample_paths_1.shape[0], size=self.__num_samples, replace=False)]
        print(sample_paths_0)
        print(sample_paths_1)
        if identity:
            self._transform(sample_paths_0, CLASS_1_SUB)
            self._transform(sample_paths_1, CLASS_2_SUB)
        self._transform(sample_paths_0, CLASS_1_SUB, sample_paths_1, CLASS_2_SUB)
        self._transform(sample_paths_1, CLASS_2_SUB, sample_paths_0, CLASS_1_SUB)

    def _transform(self, sample_paths, original_genre, dest_sample_paths=None, destination_genre=None):
        for i, sample_path in enumerate(sample_paths):
            if not self.__overlap:
                imgs, img_name = self._combine_specs_to_orig(sample_path)
            else:
                imgs, img_name = self._combine_overlapping_specs(sample_path)
            pose_codes = self.__converter.pose_encoder.predict(imgs)
            identity_codes = self.__converter.identity_embedding.predict(np.array([original_genre] * imgs.shape[0]))
            if destination_genre is not None:
                for dest_sample_path in dest_sample_paths:
                    if not self.__overlap:
                        dest_imgs, dest_img_name = self._combine_specs_to_orig(dest_sample_path)
                    else:
                        dest_imgs, dest_img_name = self._combine_overlapping_specs(dest_sample_path)
                    dest_identity_codes = self.__converter.identity_embedding.predict(np.array([destination_genre] * dest_imgs.shape[0]))
                    dest_identity_adain_params = self.__converter.identity_modulation.predict(dest_identity_codes)
                    try:
                        Path(self.__base_dir + f'/samples/' + GENRE_TRANSFORM_DIR).mkdir(parents=True)
                    except FileExistsError:
                        pass
                    if not self.__overlap:
                        converted_imgs = [
                            np.squeeze(self.__converter.generator.predict([pose_codes[[k]], dest_identity_adain_params[[k]]])[0]).T
                            for k in range(10)
                        ]
                        full_spec = np.concatenate(converted_imgs, axis=1)
                    else:
                        converted_imgs = [
                            np.pad(self.__converter.generator.predict([pose_codes[[k]], dest_identity_adain_params[[k]]])[0], pad_width=paddings, constant_values=0)
                            for k in range(13)
                        ]
                        full_spec = self._concatenate_overlap(converted_imgs)
                    imwrite(os.path.join(self.__base_dir, 'samples', GENRE_TRANSFORM_DIR, 'out-' + img_name[:-5] + '-' + str(original_genre) + '-' + str(destination_genre) + '.tif'), full_spec)
                    print(f'converting spec {img_name[:-6]}')
                    self.convert_spec_to_audio(full_spec, img_name[:-6], str(original_genre) + '-' + str(destination_genre), genre_transform=True)
            else:
                identity_adain_params = self.__converter.identity_modulation.predict(identity_codes)
                try:
                    Path(self.__base_dir + f'/samples/' + IDENTITY_TRANSFORM_DIR).mkdir(parents=True)
                except FileExistsError:
                    pass
                if not self.__overlap:
                    converted_imgs = [
                        np.pad(self.__converter.generator.predict([pose_codes[[k]], identity_adain_params[[k]]])[0], pad_width=paddings, constant_values=0)
                        for k in range(10)
                    ]
                    full_spec = np.concatenate(converted_imgs, axis=1)
                else:
                    converted_imgs = [
                        np.pad(self.__converter.generator.predict([pose_codes[[k]], identity_adain_params[[k]]])[0], pad_width=paddings, constant_values=0)
                        for k in range(13)
                    ]
                    print('length converted_imgs: ', len(converted_imgs))
                    full_spec = self._concatenate_overlap(converted_imgs)

                imwrite(os.path.join(self.__base_dir, 'samples', IDENTITY_TRANSFORM_DIR, 'out-' + img_name[:-5] + '.tif'), full_spec)
                self.convert_spec_to_audio(full_spec, img_name[:-5] + '-' + str(original_genre), genre_transform=False)

    def _concatenate_overlap(self, imgs):
        mask = binomial_mask()
        first_in_pair = np.concatenate((imgs[0], np.zeros((128, 96, 3))), axis=1)
        second_in_pair = np.concatenate((np.zeros((128, 96, 3)), imgs[1]), axis=1)
        merge = (1 - mask) * first_in_pair + mask * second_in_pair
        last_img = merge[:, -128:, :]
        full_spec = np.zeros((128, 1280, 3), dtype=np.float32)
        full_spec[:, : 2 * 128 - 32, :] = merge
        print('length imgs: ', len(imgs))
        for i, img in zip(range((128 - 32) * 2, 1280 - 128 + 1, 96), imgs[2:]):
            # print(i)
            first_in_pair = np.concatenate((last_img, np.zeros((128, 96, 3))), axis=1)
            second_in_pair = np.concatenate((np.zeros((128, 96, 3)), img), axis=1)
            merge = (1 - mask) * first_in_pair + mask * second_in_pair
            last_img = merge[:, -128:, :]
            full_spec[:, i - 96: i + 128, :] = merge
        return full_spec

    def _combine_specs_to_orig(self, sample_path):
        img_name = re.search(r'\d+\.npy', sample_path).group(0)
        img_path = os.path.join(self.__dataset_dir, 'datasets', 'fma_medium_specs_imgs', img_name)
        full_img = []
        for j in range(10):
            try:
                curr_path = img_path[:-5] + str(j) + img_path[-4:]
                img = imread(curr_path).T[:, :, None].astype(np.float32) / 255.0
                full_img.append(img)
            except FileNotFoundError:
                print('img not found at ', img_path)
                break
        return np.array(full_img), img_name

    def _combine_overlapping_specs(self, sample_path):
        img_name = re.search(r'\d+\.npy', sample_path).group(0)
        img_path = os.path.join(self.__dataset_dir, 'datasets', f'fma_medium_specs_overlap-{CLASS_1_ID}-{CLASS_2_ID}-ol', img_name)
        full_img = []
        for j in range(13):
            if j < 10:
                num = '0' + str(j)
            else:
                num = str(j)
            try:
                curr_path = img_path[:-6] + num + img_path[-4:]
                # img = imread(curr_path).T[:, :, None].astype(np.float32) / 255.0
                loaded_img = np.load(curr_path)
                img = np.zeros(loaded_img.shape)
                img[:, :, 0] = (loaded_img[:, :, 0] - default_config['min_level_db']) / (default_config['max_level_db'] - default_config['min_level_db'])
                img[:, :, 1] = (loaded_img[:, :, 1] - default_config['min_phase']) / (default_config['max_phase'] - default_config['min_phase'])
                full_img.append(img)
            except FileNotFoundError:
                print('img not found at ', img_path)
                break
        print('normalized max: ', np.max(full_img), ' min: ', np.min(full_img))
        return np.array(full_img), img_name

    def convert_spec_to_audio(self, spec, i, j=None, genre_transform=False):
        amp = (41 - (-100)) * spec[:, :, 1] + (-100)
        phase = (0.302 - (-0.303)) * spec[:, :, 2] + (-0.303)

        print('denormalized max amplitude: ', np.max(amp), ' min: ', np.min(amp))
        print('denormalized max phase: ', np.max(phase), ' min: ', np.min(phase))

        # print('denormalized: ', spec)
        amp = librosa.feature.inverse.db_to_power(amp)
        # amp = librosa.feature.inverse.mel_to_stft(amp)
        audio = librosa.feature.inverse.mel_to_audio(amp)
        # audio_orig = librosa.feature.inverse.mel_to_audio(amp_orig)
        # phase = librosa.feature.inverse.mel_to_stft(phase, power=1)
        # S = amp * np.cos(phase) + amp * np.sin(phase) * 1j

        print('performing iSTFT...')
        # audio = librosa.istft(S)
        print('iSTFT done.')

        if genre_transform:
            wavfile.write(os.path.join(self.__base_dir, 'samples', GENRE_TRANSFORM_DIR, str(i) + '-' + str(j) + '.wav'), SAMPLE_RATE, audio)
        else:
            wavfile.write(os.path.join(self.__base_dir, 'samples', IDENTITY_TRANSFORM_DIR, str(i) + '.wav'), SAMPLE_RATE, audio)


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
    mask = np.zeros((im_shape[0], 2 * im_shape[1] - int(.25 * im_shape[1]), 3))
    mask[:, im_shape[1]: im_shape[1] + int(.75 * im_shape[1]), :] = np.ones((im_shape[0], int(.75 * im_shape[1]), 3))
    mask[:, int(.75 * im_shape[1]): im_shape[1], :] = np.tile(ser_array_squash, (im_shape[0], 1, 1))
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bd', '--base-dir', type=str, required=True)
    # num_samples, model_dir, test
    parser.add_argument('-mn', '--model-name', type=str, required=True)
    parser.add_argument('-ns', '--num-samples', type=int, required=True)
    parser.add_argument('-it', '--is-test', type=int, required=True)
    parser.add_argument('-io', '--is-overlapping', type=int, required=True)
    parser.add_argument('-dd', '--dataset-dir', type=str, required=True)
    parser.add_argument('-id', '--identity', type=int, required=True)
    args = parser.parse_args()
    inferer = Inferer(args)
    inferer.infer(identity=args.identity)


if __name__ == '__main__':
    main()
