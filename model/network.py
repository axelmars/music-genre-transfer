import os
import pickle
import imageio

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import optimizers, losses, regularizers
from keras.layers import Conv2D, Dense, UpSampling2D, LeakyReLU, Activation
from keras.layers import Layer, Input, Reshape, Lambda, Flatten, Concatenate, Embedding, GaussianNoise, TimeDistributed, ConvLSTM2D
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from keras.applications import vgg16
from keras_lr_multiplier import LRMultiplier
# from keras_lr_multiplier.backend import optimizers
# from tensorflow.python.framework.errors_impl import InvalidArgumentError

from model.evaluation import EvaluationCallback, TrainEncodersEvaluationCallback


class Converter:

	class Config:

		def __init__(self, img_shape, n_imgs, n_identities,
					 pose_dim, identity_dim, pose_std, pose_decay, n_adain_layers, adain_dim,
					 perceptual_loss_layers, perceptual_loss_weights, perceptual_loss_scales):

			self.img_shape = img_shape

			self.n_imgs = n_imgs
			self.n_identities = n_identities

			self.pose_dim = pose_dim
			self.identity_dim = identity_dim

			self.pose_std = pose_std
			self.pose_decay = pose_decay

			self.n_adain_layers = n_adain_layers
			self.adain_dim = adain_dim

			self.perceptual_loss_layers = perceptual_loss_layers
			self.perceptual_loss_weights = perceptual_loss_weights
			self.perceptual_loss_scales = perceptual_loss_scales

	@classmethod
	def build(cls, img_shape, n_imgs, n_identities,
			  pose_dim, identity_dim, pose_std, pose_decay, n_adain_layers, adain_dim,
			  perceptual_loss_layers, perceptual_loss_weights, perceptual_loss_scales):

		config = Converter.Config(
			img_shape, n_imgs, n_identities,
			pose_dim, identity_dim, pose_std, pose_decay, n_adain_layers, adain_dim,
			perceptual_loss_layers, perceptual_loss_weights, perceptual_loss_scales
		)

		pose_encoder = cls.__build_pose_encoder(img_shape, pose_dim, pose_std, pose_decay)
		identity_embedding = cls.__build_identity_embedding(n_identities, identity_dim)
		identity_modulation = cls.__build_identity_modulation(identity_dim, n_adain_layers, adain_dim)
		generator = cls.__build_generator(pose_dim, n_adain_layers, adain_dim, img_shape)

		return Converter(config, pose_encoder, identity_embedding, identity_modulation, generator)

	@classmethod
	def load(cls, model_dir, include_encoders=False):
		print('loading models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		pose_encoder = load_model(os.path.join(model_dir, 'pose_encoder.h5py'))
		identity_embedding = load_model(os.path.join(model_dir, 'identity_embedding.h5py'))
		identity_modulation = load_model(os.path.join(model_dir, 'identity_modulation.h5py'))

		generator = load_model(os.path.join(model_dir, 'generator.h5py'), custom_objects={
			'AdaptiveInstanceNormalization': AdaptiveInstanceNormalization
		})

		if not include_encoders:
			return Converter(config, pose_encoder, identity_embedding, identity_modulation, generator)

		identity_encoder = load_model(os.path.join(model_dir, 'identity_encoder.h5py'))

		return Converter(config, pose_encoder, identity_embedding, identity_modulation, generator, identity_encoder)

	def save(self, model_dir):
		print('saving models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		self.pose_encoder.save(os.path.join(model_dir, 'pose_encoder.h5py'))
		self.identity_embedding.save(os.path.join(model_dir, 'identity_embedding.h5py'))
		self.identity_modulation.save(os.path.join(model_dir, 'identity_modulation.h5py'))
		self.generator.save(os.path.join(model_dir, 'generator.h5py'))

		if self.identity_encoder:
			self.identity_encoder.save(os.path.join(model_dir, 'identity_encoder.h5py'))

	def __init__(self, config,
				 pose_encoder, identity_embedding,
				 identity_modulation, generator,
				 identity_encoder=None):

		self.config = config

		self.pose_encoder = pose_encoder
		self.identity_embedding = identity_embedding
		self.identity_modulation = identity_modulation
		self.generator = generator
		self.identity_encoder = identity_encoder

		self.vgg = None
		# self.vgg = self.__build_vgg()

	def train(self, imgs, identities, batch_size, n_epochs, model_dir, tensorboard_dir):
		img = Input(shape=self.config.img_shape)
		identity = Input(shape=(1, ))

		pose_code = self.pose_encoder(img)
		identity_code = self.identity_embedding(identity)
		identity_adain_params = self.identity_modulation(identity_code)
		generated_img = self.generator([pose_code, identity_adain_params])

		model = Model(inputs=[img, identity], outputs=generated_img)

		# model.compile(
		# 	optimizer=LRMultiplier(
		# 		name='AdamOptimizer',
		# 		optimizer=optimizers.Adam(beta_1=0.5, beta_2=0.999),
		# 		multipliers={
		# 			'identity-embedding': 10.0
		# 		}
		# 	),
		#
		# 	loss=self.__l1_and_l2_loss
		# 	# loss=self.__perceptual_loss_multiscale
		# )
		model.compile(
			optimizer=optimizers.Adam(beta_1=0.5, beta_2=0.999),
			loss=self.__l1_and_l2_loss
			# loss=self.__perceptual_loss_multiscale
		)
		lr_scheduler = CosineLearningRateScheduler(max_lr=1e-4, min_lr=1e-5, total_epochs=n_epochs)
		# lr_scheduler = CosineLearningRateScheduler(max_lr=1e-4, min_lr=1e-5, total_epochs=n_epochs)
		early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=1, patience=100, verbose=1)

		tensorboard = EvaluationCallback(
			imgs, identities,
			self.pose_encoder, self.identity_embedding,
			self.identity_modulation, self.generator,
			tensorboard_dir
		)

		checkpoint = CustomModelCheckpoint(self, model_dir)

		model.fit(
			x=[imgs, identities], y=imgs,
			batch_size=batch_size, epochs=n_epochs,
			callbacks=[lr_scheduler, early_stopping, checkpoint, tensorboard],
			verbose=1
		)

	def train_identity_encoder(self, imgs, identities, batch_size, n_epochs, model_dir, tensorboard_dir):
		self.identity_encoder = self.__build_identity_encoder(self.config.img_shape, self.config.identity_dim)

		img = Input(shape=self.config.img_shape)
		pose_code = Input(shape=(self.config.pose_dim, ))

		identity_code = self.identity_encoder(img)
		identity_adain_params = self.identity_modulation(identity_code)
		generated_img = self.generator([pose_code, identity_adain_params])

		model = Model(inputs=[img, pose_code], outputs=[generated_img, identity_code])

		model.compile(
			optimizer=optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.999),
			loss=[self.__l1_and_l2_loss, losses.mean_squared_error],
			loss_weights=[1, 1e4]
		)

		reduce_lr = ReduceLROnPlateau(monitor='loss', mode='min', min_delta=1, factor=0.5, patience=20, verbose=1)
		early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=1, patience=40, verbose=1)

		tensorboard = TrainEncodersEvaluationCallback(
			imgs, self.pose_encoder, self.identity_encoder, self.identity_modulation, self.generator, tensorboard_dir
		)

		checkpoint = CustomModelCheckpoint(self, model_dir)

		model.fit(
			x=[imgs, self.pose_encoder.predict(imgs)], y=[imgs, self.identity_embedding.predict(identities)],
			batch_size=batch_size, epochs=n_epochs,
			callbacks=[reduce_lr, early_stopping, checkpoint, tensorboard],
			verbose=1
		)

	def __l1_and_l2_loss(self, y_true, y_pred):
		alpha = 0.5
		return (1-alpha) * tf.keras.losses.MeanAbsoluteError()(y_true, y_pred) + alpha * tf.keras.losses.MeanSquaredError()(y_true, y_pred)

	def __perceptual_loss(self, y_true, y_pred):
		perceptual_codes_pred = self.vgg(y_pred)
		perceptual_codes_true = self.vgg(y_true)

		normalized_weights = self.config.perceptual_loss_weights / np.sum(self.config.perceptual_loss_weights)
		loss = 0

		for i, (p, t) in enumerate(zip(perceptual_codes_pred, perceptual_codes_true)):
			loss += normalized_weights[i] * K.mean(K.abs(p - t), axis=[1, 2, 3])

		loss = K.mean(loss)
		return loss

	def __perceptual_loss_multiscale(self, y_true, y_pred):
		loss = 0

		for scale in self.config.perceptual_loss_scales:
			y_true_resized = tf.image.resize(y_true, (scale, scale), method=tf.image.ResizeMethod.BILINEAR)
			y_pred_resized = tf.image.resize(y_pred, (scale, scale), method=tf.image.ResizeMethod.BILINEAR)

			loss += self.__perceptual_loss(y_true_resized, y_pred_resized)

		return loss / len(self.config.perceptual_loss_scales)

	@classmethod
	def __build_identity_embedding(cls, n_identities, identity_dim):
		identity = Input(shape=(1, ))

		identity_embedding = Embedding(input_dim=n_identities, output_dim=identity_dim, name='identity-embedding')(identity)
		identity_embedding = Reshape(target_shape=(identity_dim, ))(identity_embedding)

		model = Model(inputs=identity, outputs=identity_embedding)

		print('identity embedding:')
		model.summary()

		return model

	@classmethod
	def __build_identity_modulation(cls, identity_dim, n_adain_layers, adain_dim):
		identity_code = Input(shape=(identity_dim, ))

		adain_per_layer = [Dense(units=adain_dim * 2)(identity_code) for _ in range(n_adain_layers)]
		adain_all = Concatenate(axis=-1)(adain_per_layer)
		identity_adain_params = Reshape(target_shape=(n_adain_layers, adain_dim, 2))(adain_all)

		model = Model(inputs=[identity_code], outputs=identity_adain_params, name='identity-modulation')

		print('identity-modulation arch:')
		model.summary()

		return model

	@classmethod
	def __build_generator(cls, pose_dim, n_adain_layers, adain_dim, img_shape):
		pose_code = Input(shape=(pose_dim, ))
		identity_adain_params = Input(shape=(n_adain_layers, adain_dim, 2))

		initial_height = img_shape[0] // (2 ** n_adain_layers)
		initial_width = img_shape[1] // (2 ** n_adain_layers)

		x = TimeDistributed(Dense(units=initial_width * (adain_dim // 8)))(pose_code)
		x = LeakyReLU()(x)

		x = TimeDistributed(Dense(units=initial_width * (adain_dim // 4)))(x)
		x = LeakyReLU()(x)

		x = TimeDistributed(Dense(units=initial_width * adain_dim))(x)
		x = LeakyReLU()(x)

		x = TimeDistributed(Reshape(target_shape=(initial_width, adain_dim)))(x)

		for i in range(n_adain_layers):
			x = UpSampling2D(size=(2, 1))(x)
			x = ConvLSTM2D(filters=adain_dim, kernel_size=(3, 1), padding='same')(x)
			x = LeakyReLU()(x)
			# x = GRU(units=512, return_sequences=True)(x)

			x = AdaptiveInstanceNormalization(adain_layer_idx=i)([x, identity_adain_params])

		x = ConvLSTM2D(filters=64, kernel_size=(5, 1), padding='same')(x)
		x = LeakyReLU()(x)

		x = ConvLSTM2D(filters=img_shape[-1], kernel_size=(7, 1), padding='same')(x)
		target_img = Activation('sigmoid')(x)

		model = Model(inputs=[pose_code, identity_adain_params], outputs=target_img, name='generator')

		print('generator arch:')
		model.summary()

		return model

	def __build_vgg(self):
		vgg = vgg16.VGG16(include_top=False, input_shape=(self.config.img_shape[0], self.config.img_shape[1], 3))

		layer_outputs = [vgg.layers[layer_id].output for layer_id in self.config.perceptual_loss_layers]
		feature_extractor = Model(inputs=vgg.inputs, outputs=layer_outputs)

		img = Input(shape=self.config.img_shape)

		if self.config.img_shape[-1] == 1:
			x = Lambda(lambda t: tf.tile(t, multiples=(1, 1, 1, 3)))(img)
		else:
			x = img

		x = VggNormalization()(x)
		features = feature_extractor(x)

		model = Model(inputs=img, outputs=features, name='vgg')

		print('vgg arch:')
		model.summary()

		return model

	@classmethod
	def __build_pose_encoder(cls, img_shape, pose_dim, pose_std, pose_decay):
		img = Input(shape=img_shape)
		# img = np.expand_dims(img)

		# x = TimeDistributed(Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same'))(img)
		# x = LeakyReLU()(x)
		# x = TimeDistributed(Flatten())(x)
		x = ConvLSTM2D(filters=64, kernel_size=(7, 1), strides=(1, 1), return_sequences=True, padding='same')(img)

		# x = Reshape(target_shape=(x_shape[0], x_shape[1], -1, 64))(x)
		# x = Conv2D(filters=128, kernel_size=(1, 4), strides=(1, 2), padding='same')(x)
		# x = LeakyReLU()(x)
		# x_shape = tf.shape(x)
		# x = Reshape(target_shape=(x_shape[0], x_shape[1], tf.reduce_prod(img_shape[2:])))(x)
		# x = GRU(units=1024, return_sequences=True)(x)
		x = ConvLSTM2D(filters=128, kernel_size=(4, 1), strides=(2, 1), return_sequences=True, padding='same')(x)

		# x = Reshape(target_shape=(x_shape[0], x_shape[1], -1, 128))(x)
		# x = Conv2D(filters=256, kernel_size=(1, 4), strides=(1, 2), padding='same')(x)
		# x = LeakyReLU()(x)
		# x_shape = tf.shape(x)
		# x = Reshape(target_shape=(x_shape[0], x_shape[1], tf.reduce_prod(img_shape[2:])))(x)
		# x = GRU(units=512, return_sequences=True)(x)
		x = ConvLSTM2D(filters=256, kernel_size=(4, 1), strides=(2, 1), return_sequences=True, padding='same')(x)

		# x = Reshape(target_shape=(x_shape[0], x_shape[1], -1, 256))(x)
		# x = Conv2D(filters=256, kernel_size=(1, 4), strides=(1, 2), padding='same')(x)
		# x = LeakyReLU()(x)
		# x_shape = tf.shape(x)
		# x = Reshape(target_shape=(x_shape[0], x_shape[1], tf.reduce_prod(img_shape[2:])))(x)
		# x = GRU(units=512, return_sequences=True)(x)
		x = ConvLSTM2D(filters=256, kernel_size=(4, 1), strides=(2, 1), return_sequences=True, padding='same')(x)

		# x = Reshape(target_shape=(x_shape[0], x_shape[1], -1, 256))(x)
		# x = Conv2D(filters=256, kernel_size=(1, 4), strides=(1, 2), padding='same')(x)
		# x = LeakyReLU()(x)
		# x_shape = tf.shape(x)
		# x = Reshape(target_shape=(x_shape[0], x_shape[1], tf.reduce_prod(img_shape[2:])))(x)
		# x = GRU(units=512, return_sequences=True)(x)
		x = ConvLSTM2D(filters=256, kernel_size=(4, 1), strides=(2, 1), return_sequences=True, padding='same')(x)

		x = TimeDistributed(Flatten())(x)

		for i in range(2):
			x = TimeDistributed(Dense(units=256))(x)
			x = LeakyReLU()(x)

		pose_code = TimeDistributed(Dense(units=pose_dim, activity_regularizer=regularizers.l2(pose_decay)))(x)
		pose_code = GaussianNoise(stddev=pose_std)(pose_code)

		model = Model(inputs=img, outputs=pose_code, name='pose-encoder')

		print('pose-encoder arch:')
		model.summary()

		return model

	@classmethod
	def __build_identity_encoder(cls, img_shape, identity_dim):
		img = Input(shape=img_shape)

		x = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same')(img)
		x = LeakyReLU()(x)

		x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		for i in range(2):
			x = Dense(units=256)(x)
			x = LeakyReLU()(x)

		identity_code = Dense(units=identity_dim)(x)

		model = Model(inputs=img, outputs=identity_code, name='identity-encoder')

		print('identity-encoder arch:')
		model.summary()

		return model


class AdaptiveInstanceNormalization(Layer):

	def __init__(self, adain_layer_idx, **kwargs):
		super().__init__(**kwargs)
		self.adain_layer_idx = adain_layer_idx

	def call(self, inputs, **kwargs):
		assert isinstance(inputs, list)

		x, adain_params = inputs
		adain_offset = adain_params[:, self.adain_layer_idx, :, 0]
		adain_scale = adain_params[:, self.adain_layer_idx, :, 1]

		adain_dim = x.shape[-1]
		adain_offset = K.reshape(adain_offset, (-1, 1, 1, adain_dim))
		adain_scale = K.reshape(adain_scale, (-1, 1, 1, adain_dim))

		mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
		x_standard = (x - mean) / (tf.sqrt(var) + 1e-7)

		return (x_standard * adain_scale) + adain_offset

	def get_config(self):
		config = {
			'adain_layer_idx': self.adain_layer_idx
		}

		base_config = super().get_config()
		return dict(list(base_config.items()) + list(config.items()))


class VggNormalization(Layer):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def call(self, inputs, **kwargs):
		x = inputs * 255
		return vgg16.preprocess_input(x)


class CustomModelCheckpoint(Callback):

	def __init__(self, model, path):
		super().__init__()
		self.__model = model
		self.__path = path

	def on_epoch_end(self, epoch, logs=None):
		self.__model.save(self.__path)


class CosineLearningRateScheduler(Callback):

	def __init__(self, max_lr, min_lr, total_epochs):
		super().__init__()

		self.max_lr = max_lr
		self.min_lr = min_lr
		self.total_epochs = total_epochs

	def on_train_begin(self, logs=None):
		K.set_value(self.model.optimizer.lr, self.max_lr)

	def on_epoch_end(self, epoch, logs=None):
		fraction = epoch / self.total_epochs
		lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction * np.pi))

		K.set_value(self.model.optimizer.lr, lr)
		logs['lr'] = lr


# class LRMultiplierWrapper(optimizers.Optimizer):
#
# 	def __init__(self, optimizer, multipliers, name, **kwargs):
# 		super(LRMultiplierWrapper, self).__init__(name, **kwargs)
# 		self.optimizer = optimizers.get(optimizer)
# 		self.multipliers = multipliers
# 		if hasattr(self.optimizer, 'learning_rate'):
# 			self.lr_attr = 'learning_rate'
# 		else:
# 			self.lr_attr = 'lr'
#
# 	@property
# 	def lr(self):
# 		return self.optimizer.lr
#
# 	@lr.setter
# 	def lr(self, lr):
# 		self.optimizer.lr = lr
#
# 	@property
# 	def learning_rate(self):
# 		return self.optimizer.learning_rate
#
# 	@learning_rate.setter
# 	def learning_rate(self, learning_rate):
# 		try:
# 			self.optimizer.learning_rate = learning_rate
# 		except ValueError:
# 			self.optimizer._hyper['learning_rate'] = learning_rate
#
# 	def _get_multiplier(self, name):
# 		multiplier, prefix_len = 1.0, 0
# 		for key, val in self.multipliers.items():
# 			if name.startswith(key):
# 				if len(key) > prefix_len:
# 					prefix_len = len(key)
# 					multiplier = val
# 		return multiplier
#
# 	def get_updates(self, loss, params):
# 		if len(self.updates) > 0:
# 			return self.updates
# 		multiplies = {}
# 		for param in params:
# 			multiplier = self._get_multiplier(param.name)
# 			if multiplier not in multiplies:
# 				multiplies[multiplier] = []
# 			multiplies[multiplier].append(param)
#
# 		self.updates, self.weights = [], []
# 		origin_lr = getattr(self, self.lr_attr)
# 		for i, (multiplier, params) in enumerate(multiplies.items()):
# 			lr = origin_lr
# 			if callable(multiplier):
# 				lr = lr * multiplier(K.cast(self.optimizer.iterations, K.floatx()))
# 			elif multiplier != 1.0:
# 				lr = lr * multiplier
# 			setattr(self, self.lr_attr, lr)
# 			with K.name_scope('Group_{}'.format(i)):
# 				self.updates += self.optimizer.get_updates(loss, params)
# 			# print(self.multipliers, i, self.optimizer.weights)
# 			print('num weights in optimizer: ', len(self.optimizer.weights))
# 			# self.optimizer.set_weights(K.cast(self.optimizer.weights, K.floatx()))
# 			# self.weights.append(self.optimizer.weights[0])
# 			for w in self.optimizer.weights:
# 				# print(w, self.weights[1:])
# 				# if w not in self.weights[1:]:
# 				# names = [x.name for x in self.weights]
# 				# if w.name not in names:
# 				# 	self.weights.append(w)
# 				#
# 				if any(w is x for x in self.weights):
# 					self.weights.append(w)
# 				# try:
# 				# if w not in self.weights:
# 				# 	self.weights.append(w)
# 				# except InvalidArgumentError:
# 				# 	if K.cast(w, K.floatx()) not in self.weights:
# 				# 		self.weights.append(K.cast(w, K.floatx()))
# 		setattr(self, self.lr_attr, origin_lr)
#
# 		return self.updates
#
# 	def get_config(self):
# 		config = {
# 			'optimizer': optimizers.serialize(self.optimizer),
# 			'multipliers': self.multipliers
# 		}
# 		base_config = super(LRMultiplierWrapper, self).get_config()
# 		return dict(list(base_config.items()) + list(config.items()))
#
# 	@classmethod
# 	def from_config(cls, config):
# 		optimizer = optimizers.deserialize(config.pop('optimizer'))
# 		return cls(optimizer, **config)

class LRMultiplierWrapper(LRMultiplier):

	def __init__(self, optimizer, multipliers, **kwargs):
		super(LRMultiplierWrapper, self).__init__(optimizer, multipliers, **kwargs)

	def get_updates(self, loss, params):
		if len(self.updates) > 0:
			return self.updates
		multiplies = {}
		for param in params:
			multiplier = self._get_multiplier(param.name)
			if multiplier not in multiplies:
				multiplies[multiplier] = []
			multiplies[multiplier].append(param)

		self.updates, self.weights = [], []
		origin_lr = getattr(self, self.lr_attr)
		for i, (multiplier, params) in enumerate(multiplies.items()):
			lr = origin_lr
			if callable(multiplier):
				lr = lr * multiplier(K.cast(self.optimizer.iterations, K.floatx()))
			elif multiplier != 1.0:
				lr = lr * multiplier
			setattr(self, self.lr_attr, lr)
			with K.name_scope('Group_{}'.format(i)):
				self.updates += self.optimizer.get_updates(loss, params)
			# print(self.multipliers, i, self.optimizer.weights)
			print('num weights in optimizer: ', len(self.optimizer.weights))
			# self.optimizer.set_weights(K.cast(self.optimizer.weights, K.floatx()))
			# self.weights.append(self.optimizer.weights[0])
			for w in self.optimizer.weights:
				# print(w, self.weights[1:])
				# if w not in self.weights[1:]:
				# names = [x.name for x in self.weights]
				# if w.name not in names:
				# 	self.weights.append(w)
				#
				if not any(w is x for x in self.weights):
					self.weights.append(w)
				# try:
				# if w not in self.weights:
				# 	self.weights.append(w)
				# except InvalidArgumentError:
				# 	if K.cast(w, K.floatx()) not in self.weights:
				# 		self.weights.append(K.cast(w, K.floatx()))
		setattr(self, self.lr_attr, origin_lr)

		return self.updates
