import io
import os

import numpy as np
from PIL import Image
from imageio import imwrite
import librosa
from scipy.io import wavfile

import tensorflow as tf
from keras.callbacks import TensorBoard

SAMPLE_RATE = 22050

class EvaluationCallback(TensorBoard):

	def __init__(self, imgs, identities, pose_encoder, identity_embedding, identity_modulation, generator, tensorboard_dir):
		super().__init__(log_dir=tensorboard_dir)
		super().set_model(generator)

		self.__imgs = imgs
		self.__identities = identities

		self.__pose_encoder = pose_encoder
		self.__identity_embedding = identity_embedding
		self.__identity_modulation = identity_modulation
		self.__generator = generator

		self.__n_samples_per_evaluation = 5
		self.writer = tf.summary.create_file_writer(tensorboard_dir)

	def on_epoch_end(self, epoch, logs={}):
		super().on_epoch_end(epoch, logs)

		img_ids = np.random.choice(self.__imgs.shape[0], size=self.__n_samples_per_evaluation, replace=False)
		imgs = self.__imgs[img_ids]
		identities = self.__identities[img_ids]

		pose_codes = self.__pose_encoder.predict(imgs)
		identity_codes = self.__identity_embedding.predict(identities)
		identity_adain_params = self.__identity_modulation.predict(identity_codes)

		blank = np.zeros_like(imgs[0])
		output = [np.concatenate([blank] + list(imgs), axis=1)]
		if not os.path.isdir('samples'):
			os.mkdir('samples')
		for i in range(self.__n_samples_per_evaluation):
			imwrite(os.path.join('samples', 'orig_img' + str(i) + '.png'), (np.squeeze(imgs[i]).T * 255).astype(np.uint8))
			convert_spec_to_audio(imgs[i], i)
			converted_imgs = [imgs[i]] + [
				self.__generator.predict([pose_codes[[j]], identity_adain_params[[i]]])[0]
				for j in range(self.__n_samples_per_evaluation)
			]
			for j in range(self.__n_samples_per_evaluation):
				img = self.__generator.predict([pose_codes[[j]], identity_adain_params[[i]]])[0]
				img = save_image(img, i, j)
				convert_spec_to_audio(img, i, j)
			output.append(np.concatenate(converted_imgs, axis=1))

		merged_img = np.concatenate(output, axis=0)
		with self.writer.as_default():
			tf.summary.image(name='sample', data=make_image(merged_img), step=epoch)
			self.writer.flush()
		# summary = tf.Summary(value=[tf.Summary.Value(tag='sample', image=make_image(merged_img))])
		# self.writer.add_summary(summary, global_step=epoch)
		# self.writer.flush()


class TrainEncodersEvaluationCallback(TensorBoard):

	def __init__(self, imgs, pose_encoder, identity_encoder, identity_modulation, generator, tensorboard_dir):
		super().__init__(log_dir=tensorboard_dir)
		super().set_model(generator)

		self.__imgs = imgs

		self.__pose_encoder = pose_encoder
		self.__identity_encoder = identity_encoder
		self.__identity_modulation = identity_modulation
		self.__generator = generator

		self.__n_samples_per_evaluation = 10
		self.writer = tf.summary.create_file_writer(tensorboard_dir)

	def on_epoch_end(self, epoch, logs={}):
		if 'loss' in logs:
			logs['loss-encoders'] = logs.pop('loss')

		if 'lr' in logs:
			logs['lr-encoders'] = logs.pop('lr')

		super().on_epoch_end(epoch, logs)

		img_ids = np.random.choice(self.__imgs.shape[0], size=self.__n_samples_per_evaluation, replace=False)
		imgs = self.__imgs[img_ids]

		pose_codes = self.__pose_encoder.predict(imgs)
		identity_codes = self.__identity_encoder.predict(imgs)
		identity_adain_params = self.__identity_modulation.predict(identity_codes)

		blank = np.zeros_like(imgs[0])
		output = [np.concatenate([blank] + list(imgs), axis=1)]
		for i in range(self.__n_samples_per_evaluation):
			converted_imgs = [imgs[i]] + [
				self.__generator.predict([pose_codes[[j]], identity_adain_params[[i]]])[0]
				for j in range(self.__n_samples_per_evaluation)
			]

			output.append(np.concatenate(converted_imgs, axis=1))

		merged_img = np.concatenate(output, axis=0)
		with self.writer.as_default():
			tf.summary.image(name='sample-with-encoders', data=make_image(merged_img), step=epoch)
			self.writer.flush()
		# summary = tf.Summary(value=[tf.Summary.Value(tag='sample-with-encoders', image=make_image(merged_img))])
		# self.writer.add_summary(summary, global_step=epoch)
		# self.writer.flush()


def make_image(tensor):
	height, width, channels = tensor.shape
	image = Image.fromarray((np.squeeze(tensor) * 255).astype(np.uint8))

	with io.BytesIO() as out:
		image.save(out, format='PNG')
		image_string = out.getvalue()

	return tf.expand_dims(tf.image.decode_png(contents=image_string, channels=channels), 0)


def save_image(tensor, im_index, pose_index):
	image = (np.squeeze(tensor).T * 255).astype(np.uint8)
	imwrite(os.path.join('samples','output' + str(im_index) + '-' + str(pose_index) + '.png'), image)
	return np.squeeze(tensor)


def convert_spec_to_audio(spec, i, j=512):
	spec = np.squeeze(spec).T
	spec = (spec * -80.0 + 80.0) * -1
	# print(spec)
	spec = librosa.feature.inverse.db_to_power(spec)
	# print('db_to_power done')
	S = librosa.feature.inverse.mel_to_stft(spec)
	print('starting griffin-lim...')
	audio = librosa.griffinlim(S)
	# audio = librosa.feature.inverse.mel_to_audio(spec, sr=22050)
	print('griffin-lim done.')
	# audio.export('reconstructed_wav.wav', format='wav')
	# audio = np.asarray(audio, dtype=np.int16)

	wavfile.write('reconstructed_wav' + str(i) + '-' + str(j) + '.wav', SAMPLE_RATE, audio)
