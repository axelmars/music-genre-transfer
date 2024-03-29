import argparse
import os
# import gc
import re

import pickle
import numpy as np

import dataset
from assets import AssetManager
from model.network import Converter
from model.network_attention import ConverterA
from config import default_config


def preprocess(args):
	assets = AssetManager(args.base_dir)

	img_dataset = dataset.get_dataset(args.dataset_id, args.dataset_path)
	imgs, identities, poses = img_dataset.read_images()
	n_identities = np.unique(identities).size

	np.savez(
		file=assets.get_preprocess_file_path(args.data_name),
		imgs=imgs, identities=identities, poses=poses, n_identities=n_identities
	)


def preprocess_genres_only(args):
	assets = AssetManager(args.base_dir)

	img_dataset = dataset.get_dataset(args.dataset_id, args.dataset_path)
	data = np.load(assets.get_preprocess_file_path(args.input_data_name))
	imgs = data['imgs']
	identities, poses = img_dataset.read_genres_only()
	n_identities = np.unique(identities).size

	np.savez(
		file=assets.get_preprocess_file_path(args.data_name),
		imgs=imgs, identities=identities, poses=poses, n_identities=n_identities
	)


def split_identities(args):
	assets = AssetManager(args.base_dir)

	data = np.load(assets.get_preprocess_file_path(args.input_data_name))
	imgs, identities, poses = data['imgs'], data['identities'], data['poses']

	n_identities = np.unique(identities).size
	test_identities = np.random.choice(n_identities, size=args.num_test_identities, replace=False)

	test_idx = np.isin(identities, test_identities)
	train_idx = ~np.isin(identities, test_identities)

	np.savez(
		file=assets.get_preprocess_file_path(args.test_data_name),
		imgs=imgs[test_idx], identities=identities[test_idx], poses=poses[test_idx], n_identities=n_identities
	)

	np.savez(
		file=assets.get_preprocess_file_path(args.train_data_name),
		imgs=imgs[train_idx], identities=identities[train_idx], poses=poses[train_idx], n_identities=n_identities
	)


def split_samples(args):
	assets = AssetManager(args.base_dir)

	data = np.load(assets.get_preprocess_file_path(args.input_data_name))
	imgs, identities, poses = data['imgs'], data['identities'], data['poses']
	print(imgs.shape)
	used_idx = ~np.all(imgs == 0, axis=(1, 2, 3))
	imgs = imgs[used_idx]
	identities = identities[used_idx]
	poses = poses[used_idx]
	print(imgs.shape)
	n_identities = np.unique(identities).size

	with open(os.path.join(args.base_dir, 'bin', 'spec_paths-ytdl.pkl'), 'rb') as f1:
		spec_paths = np.array(pickle.load(f1))

	print('spec_paths shape: ', spec_paths.shape)
	# Assuming order is kept
	spec_paths_ids = []
	for spec_path in spec_paths:
		img_name = re.search(r'\w+-\d+-\d+\.npy', spec_path).group(0)
		spec_paths_ids.append(img_name)

	spec_paths_ids = np.array(spec_paths_ids)
	spec_ids = np.unique(spec_paths_ids)
	n_samples = imgs.shape[0]
	n_test_samples = int(n_samples * args.test_split)

	random_ids = np.random.choice(np.arange(imgs.shape[0]), size=n_test_samples, replace=False)

	test_idx = random_ids
	train_idx = ~np.isin(np.arange(imgs.shape[0]), test_idx)
	# test_idx = np.nonzero(spec_paths_ids == random_ids[:, None])[1]
	print('test_idx shape: ', test_idx.shape)

	# train_idx = ~np.isin(np.arange(spec_paths_ids.shape[0]), test_idx)
	print('train_idx shape: ', np.count_nonzero(train_idx))

	# print(spec_paths_ids[train_idx][:26])
	del spec_paths_ids

	np.save(os.path.join(args.base_dir, 'bin', 'test_idx.npy'), test_idx)
	np.save(os.path.join(args.base_dir, 'bin', 'train_idx.npy'), train_idx)

	print('Saved indices')

	np.savez(
		file=assets.get_preprocess_file_path(args.test_data_name),
		imgs=imgs[test_idx], identities=identities[test_idx], poses=poses[test_idx], n_identities=n_identities
	)
	print('saved test data')

	# print('deleted ids, poses')
	# num_batches = 4
	# train_idx_int = np.arange(train_idx.shape[0])[train_idx]
	# print(f'train_idx_int shape: {train_idx_int.shape}')
	# n_in_batch = int(train_idx_int.shape[0] / num_batches)
	# print(n_in_batch)
	# for i in range(num_batches):
	# 	if i + 1 == num_batches:
	# 		indices = train_idx_int[i * n_in_batch:]
	# 	else:
	# 		indices = train_idx_int[i * n_in_batch: (i + 1) * n_in_batch]
	# 	train_imgs = imgs[indices]
	# 	print(train_imgs.shape[0])
	# 	np.savez(
	# 		file=assets.get_preprocess_file_path(str(i)), imgs=train_imgs
	# 	)
	# 	del train_imgs
	#
	# del imgs
	# imgs_train = np.zeros((train_idx_int.shape[0], 128, 128, 1), dtype=np.float32)
	# print('created zeros')
	# for i in range(num_batches):
	# 	if i + 1 == num_batches:
	# 		imgs_train[i * n_in_batch:] = np.load(assets.get_preprocess_file_path(str(i)))['imgs']
	# 	else:
	# 		imgs_train[i * n_in_batch: (i + 1) * n_in_batch] = np.load(assets.get_preprocess_file_path(str(i)))['imgs']
	#
	# print('transferred variables')
	np.savez(
		file=assets.get_preprocess_file_path(args.train_data_name),
		imgs=imgs[train_idx], identities=identities[train_idx], poses=poses[train_idx], n_identities=n_identities
	)
	print('saved train data')


def train(args):
	assets = AssetManager(args.base_dir)
	if args.resume != -1:
		model_dir = assets.get_model_dir(args.model_name)
		tensorboard_dir = assets.get_tensorboard_dir(args.model_name)
	else:
		model_dir = assets.recreate_model_dir(args.model_name)
		tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs, identities, poses, n_identities = data['imgs'], data['identities'], data['poses'], data['n_identities']

	print(np.unique(identities))
	print(np.shape(imgs))
	print(np.shape(identities))
	# for identity in np.unique(identities):
	identities[identities == 4] = 0
	identities[identities == 7] = 1
	correct_idx = (identities == 0) | (identities == 1)
	identities = identities[correct_idx]
	imgs = imgs[correct_idx]
	print(identities.shape)
	print('================= ', np.count_nonzero(identities == 1))
	print('max min amp before normalization: ', imgs[:, :, :, 0].max(), imgs[:, :, :, 0].min())
	# print('max min phase before normalization: ', imgs[:, :, :, 1:].max(), imgs[:, :, :, 1:].min())

	imgs[:, :, :, 0] = (imgs[:, :, :, 0] - default_config['min_level_db']) / (default_config['max_level_db'] - default_config['min_level_db'])
	# imgs[:, :, :, 1:] = (imgs[:, :, :, 1:] - default_config['min_phase']) / (default_config['max_phase'] - default_config['min_phase'])

	print('max amp:', imgs[:, :, :, 0].max(), 'min amp:', imgs[:, :, :, 0].min())
	# print('max phase:', imgs[:, :, :, 1:].max(), 'min phase:', imgs[:, :, :, 1:].min())
	# imgs = imgs / 255.0

	# shuffle the images
	# idx = np.arange(imgs.shape[0])
	# np.random.shuffle(idx)
	# rng = np.random.get_state()
	# np.random.shuffle(imgs)
	# np.random.set_state(rng)
	# np.random.shuffle(identities)
	# np.random.set_state(rng)
	# np.random.shuffle(poses)
	# imgs = imgs[idx]
	# identities = identities[idx]
	# poses = poses[idx]

	if args.resume != -1:
		converter = Converter.load(model_dir, include_encoders=False)

		converter.resume_train(
			imgs=imgs,
			identities=identities,

			batch_size=default_config['train']['batch_size'],
			n_epochs=default_config['train']['n_epochs'],

			model_dir=model_dir,
			tensorboard_dir=tensorboard_dir,
			resume_epoch=args.resume
		)

	else:
		converter = Converter.build(
			img_shape=imgs.shape[1:],
			n_imgs=imgs.shape[0],
			n_identities=n_identities,

			pose_dim=args.pose_dim,
			identity_dim=args.identity_dim,

			pose_std=default_config['pose_std'],
			pose_decay=default_config['pose_decay'],

			n_adain_layers=default_config['n_adain_layers'],
			adain_dim=default_config['adain_dim'],

			perceptual_loss_layers=default_config['perceptual_loss']['layers'],
			perceptual_loss_weights=default_config['perceptual_loss']['weights'],
			perceptual_loss_scales=default_config['perceptual_loss']['scales']
		)

		converter.train(
			imgs=imgs,
			identities=identities,

			batch_size=default_config['train']['batch_size'],
			n_epochs=default_config['train']['n_epochs'],

			model_dir=model_dir,
			tensorboard_dir=tensorboard_dir
		)

	converter.save(model_dir)


def train_encoders(args):
	assets = AssetManager(args.base_dir)
	if args.resume != -1:
		model_dir = assets.get_model_dir(args.model_name)
		tensorboard_dir = assets.get_tensorboard_dir(args.model_name)
	else:
		model_dir = assets.recreate_model_dir(args.model_name)
		tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs, identities, poses, n_identities = data['imgs'], data['identities'], data['poses'], data['n_identities']

	identities[identities == 4] = 0
	identities[identities == 7] = 1
	correct_idx = (identities == 0) | (identities == 1)
	identities = identities[correct_idx]
	imgs = imgs[correct_idx]

	imgs[:, :, :, 0] = (imgs[:, :, :, 0] - default_config['min_level_db']) / (default_config['max_level_db'] - default_config['min_level_db'])

	glo_backup_dir = os.path.join(model_dir, args.glo_dir)

	if args.resume != -1:
		converter = Converter.load(glo_backup_dir, include_encoders=True)
		converter.resume_train_encoders(
			imgs=imgs,
			identities=identities,
			batch_size=default_config['train']['batch_size'],
			n_epochs=default_config['train']['n_epochs'],

			model_dir=glo_backup_dir,
			tensorboard_dir=tensorboard_dir,
		)

	else:
		if not os.path.exists(glo_backup_dir):
			converter = Converter.load(model_dir, include_encoders=False)
			os.mkdir(glo_backup_dir)
			converter.save(glo_backup_dir)

		else:
			converter = Converter.load(glo_backup_dir, include_encoders=False)

		converter.train_identity_encoder(
			imgs=imgs,
			identities=identities,

			batch_size=default_config['train_encoders']['batch_size'],
			n_epochs=default_config['train_encoders']['n_epochs'],

			model_dir=model_dir,
			tensorboard_dir=tensorboard_dir
		)

	converter.save(model_dir)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bd', '--base-dir', type=str, required=True)

	action_parsers = parser.add_subparsers(dest='action')
	action_parsers.required = True

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('-di', '--dataset-id', type=str, choices=dataset.supported_datasets, required=True)
	preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=False)
	preprocess_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_parser.set_defaults(func=preprocess)

	preprocess_genres_parser = action_parsers.add_parser('preprocess-genres')
	preprocess_genres_parser.add_argument('-di', '--dataset-id', type=str, choices=dataset.supported_datasets, required=True)
	preprocess_genres_parser.add_argument('-dp', '--dataset-path', type=str, required=False)
	preprocess_genres_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_genres_parser.add_argument('-idn', '--input-data-name', type=str, required=True)
	preprocess_genres_parser.set_defaults(func=preprocess_genres_only)

	split_identities_parser = action_parsers.add_parser('split-identities')
	split_identities_parser.add_argument('-idn', '--input-data-name', type=str, required=True)
	split_identities_parser.add_argument('-trdn', '--train-data-name', type=str, required=True)
	split_identities_parser.add_argument('-tsdn', '--test-data-name', type=str, required=True)
	split_identities_parser.add_argument('-ntsi', '--num-test-identities', type=int, required=True)
	split_identities_parser.set_defaults(func=split_identities)

	split_samples_parser = action_parsers.add_parser('split-samples')
	split_samples_parser.add_argument('-idn', '--input-data-name', type=str, required=True)
	split_samples_parser.add_argument('-trdn', '--train-data-name', type=str, required=True)
	split_samples_parser.add_argument('-tsdn', '--test-data-name', type=str, required=True)
	split_samples_parser.add_argument('-ts', '--test-split', type=float, required=True)
	split_samples_parser.set_defaults(func=split_samples)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_parser.add_argument('-pd', '--pose-dim', type=int, required=True)
	train_parser.add_argument('-id', '--identity-dim', type=int, required=True)
	train_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_parser.add_argument('-ex', '--resume', type=int, default=-1)
	train_parser.add_argument('-at', '--attention', type=int, default=0)
	train_parser.set_defaults(func=train)

	train_encoders_parser = action_parsers.add_parser('train-encoders')
	train_encoders_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_encoders_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_encoders_parser.add_argument('-gd', '--glo-dir', type=str, default='glo')
	train_encoders_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_encoders_parser.set_defaults(func=train_encoders)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
