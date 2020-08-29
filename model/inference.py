import os
import numpy as np
import pickle
from imageio import imread
import re

class Inferer:
    def __init__(self, base_dir):
        self._base_dir = base_dir

    def infer(self, test=False):
        with open(os.path.join(self._base_dir, 'bin/genre_ids.pkl'), 'rb') as f2:
            genre_ids = pickle.load(f2)

        with open(os.path.join(self._base_dir, 'bin/spec_paths.pkl'), 'rb') as fd:
            spec_paths = pickle.load(fd)

        genre_ids = np.array(genre_ids)
        spec_paths = np.array(spec_paths)
        if not test:
            train_idx = np.load('bin/tran_idx.npy')
            sample_idx = np.random.choice(train_idx, size=5, replace=False)
        else:
            test_idx = np.load('bin/test_idx.npy')
            sample_idx = np.random.choice(test_idx, size=5, replace=False)
        sample_paths = spec_paths[sample_idx]
        sample_genres = genre_ids[sample_idx]
        sample_genres[sample_genres == 18] = 1
        sample_genres[sample_genres == 10] = 0
        for i, sample_path in enumerate(sample_paths):
            img_name = re.search(r'\d+\.png', sample_path).group(0)
            img_path = os.path.join(self._base_dir, 'datasets', 'fma_medium_specs_img', img_name)
            full_img = []
            for j in range(10):
                try:
                    curr_path = img_path[:-5] + str(j) + img_path[-4:]
                except FileNotFoundError:
                    break
                full_img.append(imread(curr_path).T)







