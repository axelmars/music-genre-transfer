import os
import pickle
import imageio

import numpy as np
import tensorflow as tf

from keras import backend as K

from keras import optimizers, losses, regularizers
# from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from keras.optimizers import Optimizer
from keras.layers import Conv2D, Dense, UpSampling2D, LeakyReLU, Activation
from keras.layers import Layer, Input, Reshape, Lambda, Flatten, Concatenate, Embedding, GaussianNoise
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from keras.applications import vgg16
from adamlrm import AdamLRM
from keras.utils import custom_object_scope
from keras_lr_multiplier import LRMultiplier
# from config import default_config
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
            config_dict = pickle.load(config_fd)

        with open(os.path.join(model_dir, 'optimizer.pkl'), 'rb') as opt_fd:
            opt = pickle.load(opt_fd)

        # config = config_dict['config']
        config = config_dict
        # epoch = config_dict['epoch']

        print(f'loaded optimizer with learning rate {opt.learning_rate}')

        model = load_model(os.path.join(model_dir, 'model'), custom_objects={
            'AdaptiveInstanceNormalization': AdaptiveInstanceNormalization,
            'CosineLearningRateScheduler': CosineLearningRateScheduler,
            'CustomModelCheckpoint': CustomModelCheckpoint,
            'EvaluationCallback': EvaluationCallback,
        }, compile=False)

        pose_encoder = model.layers[3]
        identity_embedding = model.layers[2]
        identity_modulation = model.layers[4]
        generator = model.layers[5]
        # pose_encoder = load_model(os.path.join(model_dir, 'pose_encoder.h5py'))
        # identity_embedding = load_model(os.path.join(model_dir, 'identity_embedding.h5py'))
        # identity_modulation = load_model(os.path.join(model_dir, 'identity_modulation.h5py'))
        #
        # generator = load_model(os.path.join(model_dir, 'generator.h5py'), custom_objects={
        #     'AdaptiveInstanceNormalization': AdaptiveInstanceNormalization
        # })

        if not include_encoders:
            return Converter(config, pose_encoder, identity_embedding, identity_modulation, generator, model, opt)

        identity_encoder = load_model(os.path.join(model_dir, 'identity_encoder.h5py'))

        return Converter(config, pose_encoder, identity_embedding, identity_modulation, generator, identity_encoder)

    def save(self, model_dir, epoch):
        print('saving models...')

        print(f'pickling config, epoch {epoch}...')
        config = {'config': self.config, 'epoch': epoch}
        with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
            pickle.dump(config, config_fd)

        print(f'pickling optimizer with learning rate {self.opt.learning_rate}...')
        with open(os.path.join(model_dir, 'optimizer.pkl'), 'wb') as opt_fd:
            pickle.dump(self.opt, opt_fd)

        # self.pose_encoder.save(os.path.join(model_dir, 'pose_encoder.h5py'))
        # self.identity_embedding.save(os.path.join(model_dir, 'identity_embedding.h5py'))
        # self.identity_modulation.save(os.path.join(model_dir, 'identity_modulation.h5py'))
        # self.generator.save(os.path.join(model_dir, 'generator.h5py'))
        print('saving model...')
        self.model.save(os.path.join(model_dir, 'model'))

        # if self.identity_encoder:
        #     self.identity_encoder.save(os.path.join(model_dir, 'identity_encoder.h5py'))

    def __init__(self, config,
                 pose_encoder, identity_embedding,
                 identity_modulation, generator, model=None, opt=None, epoch=0,
                 identity_encoder=None):

        self.config = config

        self.pose_encoder = pose_encoder
        self.identity_embedding = identity_embedding
        self.identity_modulation = identity_modulation
        self.generator = generator
        self.identity_encoder = identity_encoder
        self.model = model
        self.opt = opt
        self.epoch = epoch

        # self.vgg = None
        self.vgg = self.__build_vgg()

    def train(self, imgs, identities, batch_size, n_epochs, model_dir, tensorboard_dir):
        img = Input(shape=self.config.img_shape)
        identity = Input(shape=(1,))

        pose_code = self.pose_encoder(img)
        identity_code = self.identity_embedding(identity)
        identity_adain_params = self.identity_modulation(identity_code)
        generated_img = self.generator([pose_code, identity_adain_params])

        self.model = Model(inputs=[img, identity], outputs=generated_img)

        self.model.summary()
        # model.compile(
        #     optimizer=LRMultiplierWrapper(
        #         name='AdamOptimizer',
        #         optimizer=optimizers.Adam(beta_1=0.5, beta_2=0.999),
        #         multipliers={
        #             'identity-embedding': 10.0
        #         }
        #     ),
        #     loss=self.custom_loss
        #     # loss=self.__perceptual_loss_multiscale
        # )
        self.opt = AdamLRM(
            lr_multiplier={
                self.model.get_layer(index=2).name: 10.0
            },
            beta_1=0.5,
            beta_2=0.999
        )
        self.model.compile(
            optimizer=self.opt,
            loss=self.get_custom_loss()
        )

        # model.compile(
        #     optimizer=optimizers.Adam(beta_1=0.5, beta_2=0.999),
        #     # loss=self.__l1_and_l2_loss
        #     # loss=self.__perceptual_loss_multiscale
        #     loss=self.custom_loss
        # )

        lr_scheduler = CosineLearningRateScheduler(max_lr=1e-4, min_lr=1e-5, total_epochs=n_epochs)
        # lr_scheduler = CosineLearningRateScheduler(max_lr=1e-4, min_lr=1e-5, total_epochs=n_epochs)
        early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.01, patience=100, verbose=1)

        tensorboard = EvaluationCallback(
            imgs, identities,
            self.pose_encoder, self.identity_embedding,
            self.identity_modulation, self.generator,
            tensorboard_dir
        )
        checkpoint = CustomModelCheckpoint(self, model_dir)

        self.model.fit(
            x=[imgs, identities], y=imgs,
            batch_size=batch_size, epochs=n_epochs,
            callbacks=[lr_scheduler, early_stopping, checkpoint, tensorboard],
            verbose=1
        )

    def resume_train(self, imgs, identities, batch_size, n_epochs, model_dir, tensorboard_dir):
        self.model.summary()

        self.model.compile(
            optimizer=self.opt,
            loss=self.get_custom_loss()
        )

        lr_scheduler = CosineLearningRateScheduler(max_lr=1e-4, min_lr=1e-5, total_epochs=n_epochs, starting_epoch=self.epoch, starting_lr=self.opt.learning_rate)
        early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.01, patience=100, verbose=1)
        checkpoint = CustomModelCheckpoint(self, model_dir)

        tensorboard = EvaluationCallback(
            imgs, identities,
            self.pose_encoder, self.identity_embedding,
            self.identity_modulation, self.generator,
            tensorboard_dir
        )

        self.model.fit(
            x=[imgs, identities], y=imgs,
            batch_size=batch_size, epochs=n_epochs,
            callbacks=[lr_scheduler, early_stopping, checkpoint, tensorboard],
            verbose=1
        )

    def train_identity_encoder(self, imgs, identities, batch_size, n_epochs, model_dir, tensorboard_dir):
        self.identity_encoder = self.__build_identity_encoder(self.config.img_shape, self.config.identity_dim)

        img = Input(shape=self.config.img_shape)
        pose_code = Input(shape=(self.config.pose_dim,))

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

    def get_custom_loss(self):
        # converter = self

        def custom_loss(y_true, y_pred):
            amp_true = K.expand_dims(y_true[:, :, :, 0], axis=-1)
            phase_true = K.expand_dims(y_true[:, :, :, 1], axis=-1)

            amp_pred = K.expand_dims(y_pred[:, :, :, 0], axis=-1)
            phase_pred = K.expand_dims(y_pred[:, :, :, 1], axis=-1)

            amp_loss = self.__l1_l2_and_perceptual_loss_multiscale(amp_true, amp_pred)
            phase_loss = self.__cyclic_mse(phase_true, phase_pred)

            return 0.5 * amp_loss + 0.5 * phase_loss
        return custom_loss

    def __cyclic_mae(self, y_true, y_pred):
        return K.mean(K.abs(K.minimum(K.abs(y_true - y_pred), K.minimum(K.abs(y_pred - y_true + 1), K.abs(y_pred - y_true - 1)))), axis=-1)

    def __cyclic_mse(self, y_true, y_pred):
        return K.mean(K.square(K.minimum(K.square(y_true - y_pred), K.minimum(K.square(y_pred - y_true + 1), K.square(y_pred - y_true - 1)))), axis=-1)

    def __l1_l2_and_perceptual_loss_multiscale(self, y_true, y_pred):
        return 0.4875 * tf.keras.losses.MeanAbsoluteError()(y_true, y_pred) + 0.5 * tf.keras.losses.MeanSquaredError()(y_true, y_pred) + 0.0125 * self.__perceptual_loss_multiscale(y_true, y_pred)

    def __l1_and_l2_loss(self, y_true, y_pred):
        alpha = 0.5
        return (1 - alpha) * tf.keras.losses.MeanAbsoluteError()(y_true, y_pred) + alpha * tf.keras.losses.MeanSquaredError()(y_true, y_pred)

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
        identity = Input(shape=(1,))

        identity_embedding = Embedding(input_dim=n_identities, output_dim=identity_dim)(identity)
        identity_embedding = Reshape(target_shape=(identity_dim,))(identity_embedding)

        model = Model(inputs=identity, outputs=identity_embedding)

        print('identity embedding:')
        model.summary()

        return model

    @classmethod
    def __build_identity_modulation(cls, identity_dim, n_adain_layers, adain_dim):
        identity_code = Input(shape=(identity_dim,))

        adain_per_layer = [Dense(units=adain_dim * 2)(identity_code) for _ in range(n_adain_layers)]
        adain_all = Concatenate(axis=-1)(adain_per_layer)
        identity_adain_params = Reshape(target_shape=(n_adain_layers, adain_dim, 2))(adain_all)

        model = Model(inputs=[identity_code], outputs=identity_adain_params)

        print('identity-modulation arch:')
        model.summary()

        return model

    @classmethod
    def __build_generator(cls, pose_dim, n_adain_layers, adain_dim, img_shape):
        pose_code = Input(shape=(pose_dim,))
        identity_adain_params = Input(shape=(n_adain_layers, adain_dim, 2))

        initial_height = img_shape[0] // (2 ** n_adain_layers)
        initial_width = img_shape[1] // (2 ** n_adain_layers)

        x = Dense(units=initial_height * initial_width * (adain_dim // 8))(pose_code)
        x = LeakyReLU()(x)

        x = Dense(units=initial_height * initial_width * (adain_dim // 4))(x)
        x = LeakyReLU()(x)

        x = Dense(units=initial_height * initial_width * adain_dim)(x)
        x = LeakyReLU()(x)

        x = Reshape(target_shape=(initial_height, initial_width, adain_dim))(x)

        for i in range(n_adain_layers):
            x = UpSampling2D(size=(2, 2))(x)
            x = Conv2D(filters=adain_dim, kernel_size=(3, 3), padding='same')(x)
            x = LeakyReLU()(x)

            x = AdaptiveInstanceNormalization(adain_layer_idx=i)([x, identity_adain_params])

        x = Conv2D(filters=64, kernel_size=(5, 5), padding='same')(x)
        x = LeakyReLU()(x)

        # x = Conv2D(filters=64, kernel_size=(7, 7), padding='same')(x)
        # x = LeakyReLU()(x)

        x = Conv2D(filters=img_shape[-1], kernel_size=(7, 7), padding='same')(x)
        target_img = Activation('sigmoid')(x)

        model = Model(inputs=[pose_code, identity_adain_params], outputs=target_img)

        print('generator arch:')
        model.summary()

        return model

    def __build_vgg(self):
        vgg = vgg16.VGG16(include_top=False, input_shape=(self.config.img_shape[0], self.config.img_shape[1], 3))

        layer_outputs = [vgg.layers[layer_id].output for layer_id in self.config.perceptual_loss_layers]
        feature_extractor = Model(inputs=vgg.inputs, outputs=layer_outputs)

        img = Input(shape=(128, 128, 1))

        if self.config.img_shape[-1] == 1:
            x = Lambda(lambda t: tf.tile(t, multiples=(1, 1, 1, 3)))(img)
        elif self.config.img_shape[-1] == 2:
            x = K.expand_dims(img[:, :, :, 0], -1)
            x = Lambda(lambda t: tf.tile(t, multiples=(1, 1, 1, 3)))(x)
        # paddings = K.constant([[0, 0], [0, 0], [0, 0], [0, 1]], dtype=tf.int32)
        # channel_to_add = np.zeros(shape=(self.config.img_shape[0], self.config.img_shape[1], 1), dtype=np.float32)
        # x = Lambda(lambda t: tf.pad(t, paddings=paddings, constant_values=0))(img)
        # x = K.map_fn(lambda t: tf.concat((t, [channel_to_add]), axis=-1), img)
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

        pose_code = Dense(units=pose_dim, activity_regularizer=regularizers.l2(pose_decay))(x)
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
        self.__model.save(self.__path, epoch)


class CosineLearningRateScheduler(Callback):

    def __init__(self, max_lr, min_lr, total_epochs, starting_epoch=0, starting_lr=-1):
        super().__init__()

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.starting_epoch = starting_epoch
        if starting_lr == -1:
            self.starting_lr = self.max_lr
        else:
            self.starting_lr = starting_lr

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.starting_lr)

    def on_epoch_end(self, epoch, logs=None):
        fraction = (self.starting_epoch + 1 + epoch) / self.total_epochs
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction * np.pi))

        K.set_value(self.model.optimizer.lr, lr)
        logs['lr'] = lr

