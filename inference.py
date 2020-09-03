import os
import argparse
import numpy as np
import pickle
from imageio import imread, imwrite
import re
import librosa
from scipy.io import wavfile
# from model import evaluation
from model.network import Converter

SAMPLE_RATE = 22050


class Inferer:
    def __init__(self, args):
        # assets = AssetManager(args.base_dir)
        model_dir = os.path.join(args.base_dir, 'cache', 'models', args.model_name)
        # model_dir = assets.recreate_model_dir(args.model_name)
        self.__base_dir = args.base_dir
        self.__num_samples = args.num_samples
        self.__model_dir = model_dir
        self.__include_encoders = args.is_test

        self.__converter = Converter.load(model_dir, include_encoders=self.__include_encoders)
        self.__converter.pose_encoder.compile()

    def infer(self):
        with open(os.path.join(self.__base_dir, 'bin/genre_ids.pkl'), 'rb') as f2:
            genre_ids = pickle.load(f2)

        with open(os.path.join(self.__base_dir, 'bin/spec_paths.pkl'), 'rb') as fd:
            spec_paths = pickle.load(fd)

        genre_ids = np.array(genre_ids)
        spec_paths = np.array(spec_paths)
        genre_ids[genre_ids == 18] = 1
        genre_ids[genre_ids == 10] = 0
        if not self.__include_encoders:
            indices = np.load(os.path.join(self.__base_dir, 'bin/train_idx.npy'))
        else:
            indices = np.load(os.path.join(self.__base_dir, 'bin/test_idx.npy'))
        sample_paths = spec_paths[indices]
        sample_genres = genre_ids[indices]
        sample_paths_0 = sample_paths[sample_genres == 0]
        sample_paths_1 = sample_paths[sample_genres == 1]
        sample_paths_0 = sample_paths_0[np.random.choice(sample_paths_0.shape[0], size=self.__num_samples, replace=False)]
        sample_paths_1 = sample_paths_1[np.random.choice(sample_paths_1.shape[0], size=self.__num_samples, replace=False)]
        self._transform(sample_paths_0, 0)
        self._transform(sample_paths_1, 1)
        self._transform(sample_paths_0, 0, sample_paths_1, 1)
        self._transform(sample_paths_1, 1, sample_paths_0, 0)

    def _transform(self, sample_paths, original_genre, dest_sample_paths=None, destination_genre=None):
        for i, sample_path in enumerate(sample_paths):
            imgs, img_name = self._combine_specs_to_orig(sample_path)
            pose_codes = self.__converter.pose_encoder.predict(imgs)
            identity_codes = self.__converter.identity_embedding.predict(np.array([original_genre] * imgs.shape[0]))
            if destination_genre is not None:
                for dest_sample_path in dest_sample_paths:
                    dest_imgs, dest_img_name = self._combine_specs_to_orig(dest_sample_path)
                    dest_identity_codes = self.__converter.identity_embedding.predict(np.array([destination_genre] * dest_imgs.shape[0]))
                    dest_identity_adain_params = self.__converter.identity_modulation.predict(dest_identity_codes)
                    try:
                        os.makedirs(os.path.join(self.__base_dir, 'samples', 'genre_transform'))
                    except FileExistsError:
                        pass
                    converted_imgs = [
                        np.squeeze(self.__converter.generator.predict([pose_codes[[k]], dest_identity_adain_params[[k]]])[0]).T
                        for k in range(10)
                    ]
                    full_spec = (np.concatenate(converted_imgs, axis=1) * 255).astype(np.uint8)
                    imwrite(os.path.join(self.__base_dir, 'samples', 'genre_transform', 'out-' + img_name[:-5] + '-' + str(original_genre) + '-' + str(destination_genre) + '.png'), full_spec)
                    self.convert_spec_to_audio(full_spec, img_name[:-5], str(original_genre) + '-' + str(destination_genre), genre_transform=True)
            else:
                identity_adain_params = self.__converter.identity_modulation.predict(identity_codes)
                try:
                    os.makedirs(os.path.join(self.__base_dir, 'samples', 'identity_transform'))
                except FileExistsError:
                    pass
                converted_imgs = [
                    np.squeeze(self.__converter.generator.predict([pose_codes[[k]], identity_adain_params[[k]]])[0]).T
                    for k in range(10)
                ]
                full_spec = (np.concatenate(converted_imgs, axis=1) * 255).astype(np.uint8)
                imwrite(os.path.join(self.__base_dir, 'samples', 'identity_transform', 'out-' + img_name[:-5] + '.png'), full_spec)
                self.convert_spec_to_audio(full_spec, img_name[:-5] + '-' + str(original_genre), genre_transform=False)

    def _combine_specs_to_orig(self, sample_path):
        img_name = re.search(r'\d+\.png', sample_path).group(0)
        img_path = os.path.join(self.__base_dir, 'datasets', 'fma_medium_specs_img', img_name)
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

    def convert_spec_to_audio(self, spec, i, j=None, genre_transform=False):
        spec = (spec * -80.0 + 80.0) * -1
        spec = librosa.feature.inverse.db_to_power(spec)
        S = librosa.feature.inverse.mel_to_stft(spec)
        print('starting griffin-lim...')
        audio = librosa.griffinlim(S)
        print('griffin-lim done.')

        if genre_transform:
            wavfile.write(os.path.join(self.__base_dir, 'samples', 'genre_tranform', str(i) + '-' + str(j) + '.wav'), SAMPLE_RATE, audio)
        else:
            wavfile.write(os.path.join(self.__base_dir, 'samples', 'identity_tranform', str(i) + '.wav'), SAMPLE_RATE, audio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bd', '--base-dir', type=str, required=True)
    # num_samples, model_dir, test
    parser.add_argument('-mn', '--model-name', type=str, required=True)
    parser.add_argument('-ns', '--num-samples', type=int, required=True)
    parser.add_argument('-it', '--is-test', type=int, required=True)

    args = parser.parse_args()
    inferer = Inferer(args)
    inferer.infer()


if __name__ == '__main__':
    main()
